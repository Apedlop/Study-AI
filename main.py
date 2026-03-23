import os
import uuid
import base64
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Form, HTTPException, status, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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
OCR_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct" 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

def base64_encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- RUTAS DE NAVEGACIÓN (Corregidas para compatibilidad) ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user=Depends(get_current_user)):
    if not user: 
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"user": user}
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request=request, name="register.html")

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
    
    # httponly=True por seguridad. samesite="lax" para evitar CSRF.
    response.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# --- PROCESAMIENTO CON IA ---

@app.post("/process")
async def process_image(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Acceso no autorizado")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # PASO 1: Extracción de texto (OCR)
        ocr_result = client.chat_completion(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extrae todo el texto de esta imagen:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{base64_encode_image(file_path)}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        extracted_text = ocr_result.choices[0].message.content

        if not extracted_text or not extracted_text.strip():
            return {"success": False, "error": "No se detectó texto en la imagen."}

        # PASO 2: Generación de material de estudio
        study_prompt = f"Basado en este texto: '{extracted_text}', crea un resumen estructurado y 3 flashcards (pregunta/respuesta) en español."
        
        final_res = client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": study_prompt}]
        )

        return {
            "success": True,
            "extracted_text": extracted_text,
            "analysis": final_res.choices[0].message.content
        }

    except Exception as e:
        return {"success": False, "error": f"Error técnico: {str(e)}"}
    finally:
        # Limpieza de archivo temporal
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 es necesario para despliegues en la nube (Render/Railway)
    uvicorn.run(app, host="0.0.0.0", port=8000)