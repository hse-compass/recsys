from sqlalchemy.orm import Session
from database import SessionLocal
from models import Student, Room
import pandas as pd


def load_data():
    db: Session = SessionLocal()

    students_df = pd.read_csv("D:/RecSys/recsys/student_data/student_distribution.csv", encoding="utf-8")
    rooms_df = pd.read_csv("D:/RecSys/recsys/hostel_room_availability.csv", encoding="utf-8")

    for _, row in students_df.iterrows():
        student = Student(
            id=row["Student_ID"],
            name=row["Name"],
            age=row["Age"],
            gender=row["Gender"],
            level_of_education=row["Level_of_Higher_Education"],
            faculty=row["Faculty"],
            course=row["Course"],
            interests=row["Interests"],
            religion=row["Religion"],
            languages=row["Languages"],
            hostel_id=row["Hostel_ID"],
            room_number=row["Room_Number"],
        )
        db.add(student)

    for _, row in rooms_df.iterrows():
        room = Room(
            hostel_id=row["Hostel_ID"],
            room_number=row["Room_Number"],
            capacity=row["Capacity"],
            available_slots=row["Available_Slots"],
        )
        db.add(room)

    db.commit()
    db.close()


if __name__ == "__main__":
    load_data()
