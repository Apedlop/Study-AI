import os
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import jwt
from sqlalchemy.orm import Session
from huggingface_hub import InferenceClient

import database as db

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
SECRET_KEY = os.environ.get("SECRET_KEY", "TU_LLAVE_SECRETA_SUPER_SEGURA") 
ALGORITHM = "HS256"
# Modelos ligeros para la API de Hugging Face
VISION_MODEL = "microsoft/git-base"
TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cliente de Hugging Face (Usa la potencia de sus servidores, no los de Render)
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

# En Render, usamos /tmp para archivos temporales
UPLOAD_DIR = "/tmp"

# --- UTILIDADES ---

def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()

async def get_current_user(request: Request, database: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token: 
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user = database.query(db.User).filter(db.User.username == username).first()
        return user
    except:
        return None

# --- RUTAS ---

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

# --- AUTENTICACIÓN (Simplificada) ---

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    if database.query(db.User).filter(db.User.username == username).first():
        return HTMLResponse("Usuario ya existe. <a href='/register'>Volver</a>")
    
    hashed_pw = db.pwd_context.hash(password)
    new_user = db.User(username=username, hashed_password=hashed_pw)
    database.add(new_user)
    database.commit()
    return RedirectResponse(url="/login", status_code=302)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), database: Session = Depends(get_db)):
    user = database.query(db.User).filter(db.User.username == username).first()
    if not user or not db.pwd_context.verify(password, user.hashed_password):
        return HTMLResponse("Credenciales incorrectas. <a href='/login'>Reintentar</a>")
    
    token = jwt.encode({"sub": user.username, "exp": datetime.now(timezone.utc) + timedelta(minutes=60)}, SECRET_KEY, algorithm=ALGORITHM)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# --- PROCESAMIENTO (SIN EASYOCR) ---

@app.post("/process")
async def process_image(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not user:
        return {"success": False, "error": "No autorizado"}

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # 1. Guardar temporalmente
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 2. OCR vía API (Hugging Face hace el trabajo pesado)
        extracted_text = client.image_to_text(file_path, model=VISION_MODEL)

        if not extracted_text:
            return {"success": False, "error": "No se pudo leer la imagen."}

        # 3. Generar material de estudio
        prompt = f"Basado en este texto: '{extracted_text}', crea un resumen y 3 flashcards en español."
        final_res = client.chat_completion(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )

        return {
            "success": True,
            "extracted_text": extracted_text,
            "analysis": final_res.choices[0].message.content
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # Render asigna el puerto automáticamente en la variable PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)