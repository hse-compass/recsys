from sqlalchemy.orm import Session
from models import Student


def get_student(db: Session, student_id: int):
    return db.query(Student).filter(Student.id == student_id).first()


def get_students(db: Session):
    return db.query(Student).all()
