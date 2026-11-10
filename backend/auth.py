import os
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from backend.database import (
    initialize_database,
    get_staff_by_staff_id,
    create_staff,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

password_hash = PasswordHash.recommended()

SECRET_KEY = os.getenv("PATIENTSAFETYAI_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "Set PATIENTSAFETYAI_SECRET_KEY before starting the backend."
    )

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    staff_id: str = Field(min_length=3, max_length=30)
    email: EmailStr
    department: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    staff_id: str
    password: str


def create_access_token(staff_id: str):
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    return jwt.encode(
        {"sub": staff_id, "exp": expires},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


@router.post("/signup")
def signup(data: SignupRequest):
    initialize_database()

    try:
        hashed_password = password_hash.hash(data.password)

        create_staff(
            full_name=data.full_name.strip(),
            staff_id=data.staff_id.strip(),
            email=str(data.email).lower(),
            role="Nurse",
            department=data.department.strip(),
            password_hash=hashed_password,
        )

        return {"message": "Staff account created successfully."}

    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="Staff ID or email is already registered.",
            )
        raise


@router.post("/login")
def login(data: LoginRequest):
    initialize_database()

    staff = get_staff_by_staff_id(data.staff_id.strip())

    if not staff or not password_hash.verify(
        data.password,
        staff["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid Staff ID or password.",
        )

    token = create_access_token(staff["staff_id"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "staff": {
            "full_name": staff["full_name"],
            "staff_id": staff["staff_id"],
            "email": staff["email"],
            "role": staff["role"],
            "department": staff["department"],
        },
    }