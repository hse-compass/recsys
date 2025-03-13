from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base
from crud import get_students, get_student
from recommendations import calculate_recommendations, recommend_students
from schemas import StudentOut
from fastapi.middleware.cors import CORSMiddleware

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/recommendations/{student_id}", response_model=list[StudentOut])
def get_recommendations(student_id: int, db: Session = Depends(get_db)):
    students = get_students(db)
    if not students:
        raise HTTPException(status_code=404, detail="Students not found")

    df, similarity_matrix = calculate_recommendations(students)
    recommendations = recommend_students(student_id, students, similarity_matrix, df)

    return recommendations
