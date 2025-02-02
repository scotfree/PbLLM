from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import openai
import jwt
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from collections import defaultdict

app = FastAPI()
Base = declarative_base()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password_hash = Column(String)

class Turn(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(String)
    date = Column(DateTime, default=datetime.utcnow)
    is_ai_generated = Column(Boolean, default=False)

class Story(Base):
    __tablename__ = "stories"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///./game.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# Auth setup
SECRET_KEY = "your-secret-key"  # Change this in production
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Add after FastAPI initialization
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# OAuth2 configuration
config = Config('.env')
oauth = OAuth(config)

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'

oauth.register(
    name='google',
    server_metadata_url=GOOGLE_CONF_URL,
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={
        'scope': 'openid email profile'
    }
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except:
        raise HTTPException(status_code=401)

# Game logic
def generate_ai_turn(user_id: int, db: Session):
    # Get user's previous turns
    previous_turns = db.query(Turn).filter(
        Turn.user_id == user_id
    ).order_by(Turn.date.desc()).limit(5).all()
    
    previous_texts = [turn.text for turn in previous_turns]
    prompt = f"Based on these previous turns: {previous_texts}, write a new short turn (max 100 words) in a similar style:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_text = response.choices[0].message.content
    
    new_turn = Turn(
        user_id=user_id,
        text=ai_text,
        is_ai_generated=True
    )
    db.add(new_turn)
    db.commit()

def should_generate_story(db: Session):
    """Check if all users have submitted their turns for today"""
    today = datetime.utcnow().date()
    all_users = db.query(User).all()
    today_turns = db.query(Turn).filter(
        Turn.date >= today
    ).all()
    
    return len(today_turns) == len(all_users)

def generate_combined_story(turns: List[Turn]) -> str:
    """Use ChatGPT to generate a cohesive story from individual turns"""
    turns_text = "\n".join([f"Turn {i+1}: {turn.text}" for i, turn in enumerate(turns)])
    
    prompt = f"""Given these story turns from different players:

{turns_text}

Create a cohesive story that incorporates all these elements naturally. 
Keep the core ideas from each turn but make them flow together smoothly.
Limit the response to 500 words."""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def combine_daily_story():
    with SessionLocal() as db:
        today = datetime.utcnow().date()
        today_turns = db.query(Turn).filter(
            Turn.date >= today
        ).order_by(Turn.date).all()
        
        # Generate missing turns
        all_users = db.query(User).all()
        for user in all_users:
            user_turn = next((t for t in today_turns if t.user_id == user.id), None)
            if not user_turn:
                generate_ai_turn(user.id, db)
        
        # Get final list of turns including AI-generated ones
        final_turns = db.query(Turn).filter(
            Turn.date >= today
        ).order_by(Turn.date).all()
        
        # Generate combined story using ChatGPT
        combined_text = generate_combined_story(final_turns)
        
        story = Story(text=combined_text)
        db.add(story)
        db.commit()

# Schedule daily story combination
scheduler = BackgroundScheduler()
scheduler.add_job(combine_daily_story, 'cron', hour=0)
scheduler.start()

# API endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or user.password_hash != form_data.password:  # Use proper password hashing in production
        raise HTTPException(status_code=400)
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/turn")
async def submit_turn(text: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if user already submitted today
    today = datetime.utcnow().date()
    existing_turn = db.query(Turn).filter(
        Turn.user_id == current_user.id,
        Turn.date >= today
    ).first()
    
    if existing_turn:
        raise HTTPException(status_code=400, detail="Already submitted today")
    
    new_turn = Turn(user_id=current_user.id, text=text)
    db.add(new_turn)
    db.commit()
    
    # Check if all users have submitted their turns
    if should_generate_story(db):
        combine_daily_story()
        return {"message": "Turn submitted successfully and story generated"}
    
    return {"message": "Turn submitted successfully"}

@app.get("/stories")
async def get_stories(db: Session = Depends(get_db)):
    stories = db.query(Story).order_by(Story.date.desc()).limit(10).all()
    return stories

@app.get('/login/google')
async def google_login(request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/auth')
async def auth(request):
    token = await oauth.google.authorize_access_token(request)
    user_info = await oauth.google.parse_id_token(request, token)
    
    # Check if user exists, if not create new user
    db = SessionLocal()
    user = db.query(User).filter(User.username == user_info['email']).first()
    
    if not user:
        user = User(
            username=user_info['email'],
            password_hash='sso_user'  # Special marker for SSO users
        )
        db.add(user)
        db.commit()
    
    # Create access token
    access_token = create_access_token({"sub": user.username})
    
    # Return HTML that sets token and closes window
    return HTMLResponse('''
        <script>
            localStorage.setItem('token', '{}');
            window.opener.postMessage('login_success', '*');
            window.close();
        </script>
    '''.format(access_token)) 