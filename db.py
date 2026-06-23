"""
Слой хранения результатов обследований.

На Railway используется managed Postgres (переменная окружения DATABASE_URL).
Локально, если DATABASE_URL не задана, используется файл SQLite (assessments.db)
— чтобы приложение работало и без базы при разработке.
"""

import os
import json
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker


def _database_url() -> str:
    """Возвращает URL подключения. Фолбэк на локальный SQLite."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return "sqlite:///assessments.db"
    # Railway/Heroku иногда отдают устаревшую схему postgres:// — SQLAlchemy её не принимает
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


DATABASE_URL = _database_url()
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Assessment(Base):
    """Одно завершённое обследование военнослужащего."""

    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(64), index=True)
    full_name = Column(String(255), index=True)
    military_recommendation = Column(String(64))

    # Сырые данные сессии хранятся как JSON (JSONB в Postgres)
    questionnaire = Column(JSON)
    responses = Column(JSON)
    scale_scores = Column(JSON)
    risk_levels = Column(JSON)
    risk_levels_desc = Column(JSON)
    detailed_results = Column(JSON)
    final_recommendations = Column(JSON)


def init_db() -> None:
    """Создаёт таблицы, если их ещё нет. Идемпотентно."""
    Base.metadata.create_all(engine)


def _jsonable(obj):
    """Приводит вложенные структуры к JSON-безопасному виду (даты -> строки и т.п.)."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def save_assessment(payload: dict) -> int:
    """Сохраняет результат обследования и возвращает его id."""
    init_db()
    db = SessionLocal()
    try:
        questionnaire = payload.get("questionnaire_responses") or {}
        row = Assessment(
            session_id=payload.get("session_id"),
            full_name=questionnaire.get("full_name"),
            military_recommendation=payload.get("military_recommendation"),
            questionnaire=_jsonable(questionnaire),
            responses=_jsonable(payload.get("responses")),
            scale_scores=_jsonable(payload.get("scale_scores")),
            risk_levels=_jsonable(payload.get("risk_levels")),
            risk_levels_desc=_jsonable(payload.get("risk_levels_desc")),
            detailed_results=_jsonable(payload.get("detailed_results")),
            final_recommendations=_jsonable(payload.get("final_recommendations")),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()


def list_assessments(limit: int = 200):
    """Список последних обследований (свежие сверху)."""
    init_db()
    db = SessionLocal()
    try:
        return (
            db.query(Assessment)
            .order_by(Assessment.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()


def get_assessment(assessment_id: int):
    """Одно обследование по id."""
    init_db()
    db = SessionLocal()
    try:
        return db.get(Assessment, assessment_id)
    finally:
        db.close()
