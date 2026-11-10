import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent.parent / "patientsafetyai.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS staff (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                staff_id TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                role TEXT NOT NULL DEFAULT 'Nurse',
                department TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()


def get_staff_by_staff_id(staff_id):
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM staff WHERE staff_id = ?",
            (staff_id,)
        ).fetchone()


def create_staff(
    full_name,
    staff_id,
    email,
    role,
    department,
    password_hash
):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO staff (
                full_name,
                staff_id,
                email,
                role,
                department,
                password_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            full_name,
            staff_id,
            email,
            role,
            department,
            password_hash
        ))

        connection.commit()
        return cursor.lastrowid