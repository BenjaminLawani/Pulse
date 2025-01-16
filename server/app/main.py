import io
import uuid
import random
import string
import re
import torch
from config import *
import os
from datetime import datetime, timedelta, UTC
from passlib.context import CryptContext
from jwt import encode, decode, PyJWTError
import numpy as np
import pydicom
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field
from starlette.requests import Request
from groq import Groq

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID
from services import EmailService, EmailServiceConfig
from slowapi import Limiter
from slowapi.util import get_remote_address
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(engine, autocommit=False)
limiter = Limiter(key_func=get_remote_address)
Base = declarative_base()


import warnings
warnings.filterwarnings("ignore")

groq_client = Groq(
    api_key= GROQ_API_KEY
)
templates = Jinja2Templates("../templates")
email_config = EmailServiceConfig(
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_FROM=MAIL_FROM,
    MAIL_PORT=587,
    MAIL_SERVER=MAIL_SERVER,
    MAIL_FROM_NAME=MAIL_FROM_NAME,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

fm = EmailService(config=email_config)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(32), nullable=True, unique=True, index=True)
    password = Column(String(64), nullable=False)
    email = Column(String, nullable=False, unique=True)
    hospital_name = Column(String, nullable=True, unique=False)

Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password): 
    return pwd_context.hash(password)

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)

def jwt_encode(data:dict):
    """
    Create JWT Token
    """
    return encode(data, os.getenv("JWT_KEY"), algorithm="HS256")

def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Create Access Token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=30) # No magic numbers ode!.
    to_encode.update({"exp": expire})
    encoded_jwt = jwt_encode(to_encode)
    return encoded_jwt

def generate_temporary_password(length: int = 8) -> str:
    """
    Generate a temporary password consisting of random letters and digits.

    Parameters:
    length (int): The length of the generated password. Default is 8.

    Returns:
    str: A randomly generated password containing uppercase letters, lowercase letters, and digits.

    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


class UserBase(BaseModel):
    username: str
    email: EmailStr
    hospital_name: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
class ReportRequest(BaseModel):
    findings: list
    patient_info: dict

class ReportResponse(BaseModel):
    raw_report: str
    structured_report: dict

class UserLogin(BaseModel):
    email: str = Field(..., description="Email or username of the user")
    password: str = Field(..., min_length=8)

class UserForgotPassword(BaseModel):
    email: EmailStr

class RadiologyAssistant:
    def __init__(self):
        # Pre-trained models for different imaging modalities
        self.brain_tumor_model = self.load_brain_tumor_model()
        self.bone_fracture_model = self.load_bone_fracture_model()

    def load_brain_tumor_model(self):
        """
        Load a pre-trained brain tumor classification model
        Uses transfer learning on brain MRI datasets
        """
        model_path = "Devarshi/Brain_Tumor_Classification"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        return {
            "model": model,
            "feature_extractor": feature_extractor,
            "conditions": ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
        }

    def load_bone_fracture_model(self):
        """
        Load a pre-trained bone fracture detection model
        Uses transfer learning on fracture X-ray datasets
        """
        model_path = "Heem2/bone-fracture-detection-using-xray"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        return {
            "model": model,
            "feature_extractor": feature_extractor,
            "conditions": ["Normal", "Fracture"]
        }

    def convert_dicom_to_image(self, dicom_data):
        """
        Convert DICOM file to PIL Image
        
        Args:
            dicom_data (pydicom.Dataset): DICOM file data
        
        Returns:
            PIL.Image: Converted image
        """
        pixel_array = dicom_data.pixel_array
    
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack((pixel_array,)*3, axis=-1)
        
        # Convert to PIL Image
        return Image.fromarray(pixel_array)

    def analyze_medical_image(self, image, modality='brain'):
        """
        Analyze medical images across different modalities
        Supports both standard image formats and DICOM
        """
        if modality == 'brain':
            model_config = self.brain_tumor_model
        elif modality == 'bone':
            model_config = self.bone_fracture_model
        else:
            raise ValueError("Unsupported imaging modality")

        # Preprocess image
        inputs = model_config['feature_extractor'](image, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = model_config['model'](**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
        
        # Process results
        results = []
        for prob, condition in zip(probabilities, model_config['conditions']):
            if prob.item() > 0.5:  # Threshold for significant findings
                results.append({
                    "condition": condition,
                    "probability": float(prob.item())
                })
        
        return results

class DiagnosticRequest(BaseModel):
    modality: str = 'brain'

class DiagnosticResult(BaseModel):
    findings: list
    patient_info: dict = {}

app = FastAPI(
    title="Pulse API",
    description="AI-powered radiography assistant",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="../templates/static"), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize RadiologyAssistant
radiology_assistant = RadiologyAssistant()

@app.post("/analyze/image", response_model=DiagnosticResult)
async def analyze_image(
    file: UploadFile = File(...), 
    modality: str = 'brain'
):
    # Validate modality
    if modality not in ['brain', 'bone']:
        raise HTTPException(status_code=400, detail="Invalid modality. Choose 'brain' or 'bone' as only those are handled right now.")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Determine image type
        if file.filename.lower().endswith(('.dcm', '.dicom')):
            # Process DICOM file
            dicom_data = pydicom.dcmread(io.BytesIO(contents))
            image = radiology_assistant.convert_dicom_to_image(dicom_data)
            
            # Extract patient info
            patient_info = {
                "Patient Name": str(dicom_data.get('PatientName', 'Unknown')),
                "Patient ID": str(dicom_data.get('PatientID', 'Unknown')),
                "Study Date": str(dicom_data.get('StudyDate', 'Unknown')),
                "Modality": str(dicom_data.get('Modality', modality.upper()))
            }
        else:
            # Process standard image file
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            patient_info = {}
    
        # Analyze image
        findings = radiology_assistant.analyze_medical_image(image, modality)
        
        return {
            "findings": findings,
            "patient_info": patient_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/get-started')
def get_create_account_page(request: Request):
    """Render the account creation page"""
    return templates.TemplateResponse('get-started.html', {'request': request})

@app.get('/login')
def login_page(request: Request):
    """Render the about page"""
    return templates.TemplateResponse('login.html', {'request': request})

@app.get('/upload')
def get_upload_page(request: Request):
    """Render the account creation page"""
    return templates.TemplateResponse('upload.html', {'request': request})

@app.post("/generate-report")
async def generate_radiology_report(
    request: Request,
    report_request: ReportRequest
):
    try:
        findings_str = "\n".join(report_request.findings)
        patient_info_str = "\n".join([f"{k}: {v}" for k, v in report_request.patient_info.items()])
        
        prompt = f"""Generate a comprehensive, professional radiology report with the following details:

Patient Information:
{patient_info_str}

Imaging Findings:
{findings_str}

Please create a structured report that includes:
1. A concise summary of patient information
2. Detailed description of findings
3. Preliminary interpretation
4. Recommendations for follow-up or additional investigation
5. Use clear, medical-professional language
6. Avoid speculative statements
7. Maintain patient confidentiality

Format the report to be clear, precise, and clinically relevant."""

        # Generate report using Groq
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional radiologist assistant generating detailed medical reports."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,  
            max_tokens=1024
        )
        
        # Extract the generated report
        raw_report = chat_completion.choices[0].message.content
        
        # Basic structured report parsing (can be enhanced)
        structured_report = {
            "patient_info": report_request.patient_info,
            "findings": report_request.findings,
            "interpretation": _extract_section(raw_report, "Interpretation"),
            "recommendations": _extract_section(raw_report, "Recommendations")
        }
        
        return {
            "raw_report": raw_report,
            "structured_report": structured_report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

def _extract_section(report: str, section_name: str) -> str:
    """
    Simple helper to extract a specific section from the report.
    
    Args:
        report (str): Full report text
        section_name (str): Name of section to extract
    
    Returns:
        str: Extracted section text or empty string
    """
    try:
        # Basic section extraction logic
        section_start = report.lower().find(section_name.lower())
        if section_start == -1:
            return ""
        
        # Find next section or end of document
        remaining_text = report[section_start:]
        next_section_match = re.search(r'\n\d+\.|\n[A-Z]', remaining_text[len(section_name):])
        
        if next_section_match:
            section_end = section_start + len(section_name) + next_section_match.start()
            return report[section_start:section_end].strip()
        else:
            return remaining_text.strip()
    except Exception:
        return ""
    
@app.post('/get-started', response_model=UserResponse)
async def create_account(
    request: Request,
    user: UserCreate, 
    db: Session = Depends(get_db)
):
    """Create a new user account"""
    try:
        existing_email = db.query(User).filter(User.email == user.email).first()
        if existing_email:
            raise HTTPException(status_code=409, detail="Email already exists")

        existing_username = db.query(User).filter(User.username == user.username).first()
        if existing_username:
            raise HTTPException(status_code=409, detail="Username already exists")
        
        new_user = User(
            username=user.username,
            email=user.email,
            password=hash_password(user.password)
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return UserResponse(
            id=str(new_user.id),
            username=new_user.username,
            email=new_user.email
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create account: {str(e)}")

@app.post('/login')
async def sign_in(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
    """Handle user login"""
    try:
        db_user = db.query(User).filter(
            (User.email == user_data.email) | (User.username == user_data.email)
        ).first()
        
        if not db_user or not verify_password(user_data.password, db_user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        

        access_token = create_access_token(
            data={
                "sub": db_user.email,
                "user_id": str(db_user.id),
                "username": db_user.username
            }
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Login failed due to server error"
        )

@app.get("/forgot-password")
async def get_forgot_password_page(request: Request):
    return templates.TemplateResponse('forgot-password.html', {'request': request})

@app.post('/forgot-password')
async def forgot_password(request: Request, user: UserForgotPassword, db: Session = Depends(get_db)):
    """Handle password reset requests"""
    if not user.email:
        raise HTTPException(
            status_code=400,
            detail="Email is required"
        )
        
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        
        if not db_user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        temp_password = generate_temporary_password()
        hashed_password = hash_password(temp_password)
        
        db_user.password = hashed_password
        db.commit()

        await fm.send_email(
            subject="Password Reset Request",
            recipients=[db_user.email],
            body=f"""
                <h2>Password Reset</h2>
                <p>Your temporary password is: <strong>{temp_password}</strong></p>
                <p>Please log in and change your password immediately.</p>
                <p>If you didn't request this password reset, please contact support immediately.</p>
            """,
            subtype="html"
        )
        
        return {"message": "A temporary password has been sent to your email."}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to process password reset request"
        )
