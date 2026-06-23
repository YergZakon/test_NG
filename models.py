"""Модели данных: пользователи (военнослужащие/психологи), назначения, обследования."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    login = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)          # 'soldier' | 'psychologist'
    full_name = Column(String(255), nullable=False)
    rank = Column(String(64))
    unit = Column(String(64))

    @property
    def initials(self) -> str:
        parts = (self.full_name or "").split()
        if not parts:
            return "—"
        ini = parts[0][:1]
        if len(parts) > 1:
            ini += parts[1][:1]
        return ini.upper()


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String(255), default="Комплексное психологическое обследование")
    title_kz = Column(String(255), default="Кешенді психологиялық тексеру")
    assigned_by = Column(String(255))
    due = Column(String(64))
    active = Column(Boolean, default=True)

    user = relationship("User")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    status = Column(String(20), default="in_progress")   # in_progress | completed

    # снимок данных тестируемого (на момент прохождения)
    full_name = Column(String(255))
    rank = Column(String(64))
    unit = Column(String(64))

    # ход теста
    stage = Column(String(20), default="anketa")         # anketa | screening | deepening | done
    questionnaire = Column(JSON, default=dict)
    responses = Column(JSON, default=dict)
    deep_plan = Column(JSON, default=list)               # [{scale, level}]
    cursor = Column(JSON, default=dict)                  # {screening_idx, block_idx, q_idx}

    # результаты
    scale_scores = Column(JSON, default=dict)
    risk_levels = Column(JSON, default=dict)
    risk_levels_desc = Column(JSON, default=dict)
    detailed_results = Column(JSON, default=dict)
    npu = Column(Integer)
    risk_label = Column(String(20))                      # norma | medium | high | progress
    military_recommendation = Column(String(40))
    final_messages = Column(JSON, default=list)

    user = relationship("User")
