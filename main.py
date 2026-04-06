import warnings
import os
import uuid
import time
import easyocr
from PIL import Image
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# 1. Silenciar avisos técnicos de librerías para un log limpio
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from huggingface_hub import InferenceClient

import database as db

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
SECRET_KEY = os.environ.get("SECRET_KEY", "TU_LLAVE_SECRETA_SUPER_SEGURA") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Usamos el modelo 7B para máxima velocidad en la API gratuita
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct" 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Inicializar EasyOCR (Procesamiento Local - Más rápido y fiable que subir la imagen)
# Esto descarga los modelos la primera vez que se ejecuta
reader = easyocr.Reader(['es', 'en'], gpu=False) 

# Cliente de Hugging Face
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- UTILIDADES ---

def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, database: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token: 
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = database.query(db.User).filter(db.User.username == username).first()
        return user
    except JWTError:
        return None

# --- RUTAS DE NAVEGACIÓN ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user=Depends(get_current_user)):
    if not user: 
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        name="index.html", 
        context={"request": request, "user": user}
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(name="login.html", context={"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(name="register.html", context={"request": request})

# --- ACCIONES DE AUTENTICACIÓN ---

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    existing_user = database.query(db.User).filter(db.User.username == username).first()
    if existing_user:
        return HTMLResponse("El usuario ya existe. <a href='/register'>Volver</a>", status_code=400)
    
    hashed_pw = db.pwd_context.hash(password)
    new_user = db.User(username=username, hashed_password=hashed_pw)
    database.add(new_user)
    database.commit()
    return RedirectResponse(url="/login", status_code=302)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    user = database.query(db.User).filter(db.User.username == username).first()
    if not user or not db.pwd_context.verify(password, user.hashed_password):
        return HTMLResponse("Credenciales incorrectas. <a href='/login'>Reintentar</a>", status_code=401)
    
    token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# --- PROCESAMIENTO HÍBRIDO (OCR Local + IA Nube) ---

@app.post("/process")
async def process_image(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not user:
        return {"success": False, "error": "Inicia sesión para procesar."}

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # 1. Guardar y optimizar imagen para OCR
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        with Image.open(file_path) as img:
            if img.width > 1200:
                img.thumbnail((1200, 1200))
                img.save(file_path, optimize=True, quality=85)

        # 2. OCR LOCAL (Infalible)
        print(f"--- Iniciando OCR Local ---")
        results = reader.readtext(file_path, detail=0, paragraph=True)
        extracted_text = "\n".join(results)

        if not extracted_text.strip():
            return {"success": False, "error": "No se detectó texto en la imagen."}

        # 3. IA PARA ANÁLISIS (Solo enviamos el texto extraído)
        print(f"--- Analizando con {LLM_MODEL} ---")
        prompt = (
            f"Basado exclusivamente en este texto extraído de mis apuntes: '{extracted_text}'\n\n"
            "Tarea: Genera un resumen estructurado y 3 flashcards (Pregunta/Respuesta) en español."
        )

        final_res = client.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return {
            "success": True,
            "extracted_text": extracted_text, # Todo el texto sin recortes
            "analysis": final_res.choices[0].message.content
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"success": False, "error": f"Error técnico: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)