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

load_dotenv()

# --- CONFIGURACIÓN ---
SECRET_KEY = os.environ.get("SECRET_KEY", "una_clave_muy_secreta_123") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
OCR_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct" 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Intentar montar carpeta static si existe (opcional)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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

# --- RUTAS DE NAVEGACIÓN ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user=Depends(get_current_user)):
    if not user: 
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# --- ACCIONES AUTH ---

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    existing_user = database.query(db.User).filter(db.User.username == username).first()
    if existing_user:
        return HTMLResponse("El usuario ya existe. <a href='/register'>Volver</a>", status_code=400)
    
    hashed_pw = db.pwd_context.hash(password)
    new_user = db.User(username=username, hashed_password=hashed_pw)
    database.add(new_user)
    database.commit()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    user = database.query(db.User).filter(db.User.username == username).first()
    if not user or not db.pwd_context.verify(password, user.hashed_password):
        return HTMLResponse("Credenciales incorrectas. <a href='/login'>Reintentar</a>", status_code=401)
    
    token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    
    # httponly=True evita robos de token por JS
    # samesite="lax" es estándar para navegación
    response.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# --- LÓGICA DE IA ---

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

        # 1. OCR con Visión-Lenguaje
        ocr_result = client.chat_completion(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrae todo el texto de esta imagen de forma literal:"},
                    {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{base64_encode_image(file_path)}"}}
                ]
            }],
            max_tokens=500
        )
        
        extracted_text = ocr_result.choices[0].message.content

        if not extracted_text or not extracted_text.strip():
            return {"success": False, "error": "No se detectó texto claro en la imagen."}

        # 2. Resumen Académico
        study_prompt = f"Resume este contenido educativo y genera 3 preguntas de estudio: '{extracted_text}'"
        
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
        return {"success": False, "error": f"Error procesando la imagen: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)