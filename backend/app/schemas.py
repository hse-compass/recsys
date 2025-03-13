from pydantic import BaseModel
from typing import Optional


class StudentBase(BaseModel):
    name: str
    age: int
    gender: str
    level_of_education: str
    faculty: str
    course: int
    interests: str
    religion: str
    languages: str
    hostel_id: Optional[int] = None
    room_number: Optional[int] = None


class StudentCreate(StudentBase):
    pass


class StudentOut(StudentBase):
    id: int

    class Config:
        orm_mode = True
