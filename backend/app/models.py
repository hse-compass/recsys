from sqlalchemy import Column, Integer, String
from database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    level_of_education = Column(String)
    faculty = Column(String)
    course = Column(Integer)
    interests = Column(String)
    religion = Column(String)
    languages = Column(String)
    hostel_id = Column(Integer)
    room_number = Column(Integer)


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    hostel_id = Column(Integer)
    room_number = Column(Integer)
    capacity = Column(Integer)
    available_slots = Column(Integer)

