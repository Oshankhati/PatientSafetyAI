import sqlite3
import json
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
        connection.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                settings_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS event_acknowledgements (
                event_id TEXT PRIMARY KEY,
                staff_id TEXT NOT NULL,
                acknowledged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def get_app_settings():
    with get_connection() as connection:
        row = connection.execute(
            "SELECT settings_json FROM app_settings WHERE id = 1"
        ).fetchone()
    return None if row is None else json.loads(row["settings_json"])


def save_app_settings(settings):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO app_settings (id, settings_json, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (json.dumps(settings),),
        )
        connection.commit()


def get_event_acknowledgements():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT a.event_id, a.staff_id, s.full_name, a.acknowledged_at
            FROM event_acknowledgements AS a
            LEFT JOIN staff AS s ON s.staff_id = a.staff_id
            """
        ).fetchall()
    return {
        row["event_id"]: {
            "staff_id": row["staff_id"],
            "staff_name": row["full_name"] or row["staff_id"],
            "acknowledged_at": row["acknowledged_at"],
        }
        for row in rows
    }


def acknowledge_event(event_id, staff_id):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO event_acknowledgements (event_id, staff_id)
            VALUES (?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (event_id, staff_id),
        )
        row = connection.execute(
            """
            SELECT a.event_id, a.staff_id, s.full_name, a.acknowledged_at
            FROM event_acknowledgements AS a
            LEFT JOIN staff AS s ON s.staff_id = a.staff_id
            WHERE a.event_id = ?
            """,
            (event_id,),
        ).fetchone()
        connection.commit()
    return {
        "event_id": row["event_id"],
        "staff_id": row["staff_id"],
        "staff_name": row["full_name"] or row["staff_id"],
        "acknowledged_at": row["acknowledged_at"],
    }