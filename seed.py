"""Инициализация БД и демо-данные. Идемпотентно: повторный запуск не дублирует."""
from datetime import datetime, timedelta

from database import engine, Base, SessionLocal
from models import User, Assignment, Assessment
from auth import hash_password
import scoring

SCALE_KEYS = list(scoring.SCALE_NAMES.keys())


def _scores_for(label: str):
    """Сгенерировать правдоподобные баллы шкал и описания рисков под целевой уровень."""
    scores, desc = {}, {}
    for k in SCALE_KEYS:
        if k == "sincerity":
            scores[k] = 7
            desc[k] = "достоверно"
            continue
        scores[k] = 5
        desc[k] = "низкий уровень риска"
    if label == "medium":
        scores["anxiety"] = 10
        desc["anxiety"] = "средний уровень риска (требуется дополнительная оценка)"
    elif label == "high":
        scores["aggression"] = 13
        desc["aggression"] = "высокий уровень риска (подтверждено углубленной оценкой)"
    return scores, desc


def _npu_for(label: str) -> int:
    return {"norma": 8, "medium": 5, "high": 3}.get(label, 8)


DEMO = [
    ("Абдрахманов К.М.", "Ст.серж.", "12-я МСБ", "high", "completed"),
    ("Нурланов Б.Е.", "Капитан", "5-й ОДО", "norma", "completed"),
    ("Смагулов Д.Н.", "Сержант", "3-я ТБ", "medium", "completed"),
    ("Жаксыбеков А.Т.", "Лейтенант", "7-я ОГБ", None, "in_progress"),
    ("Бекова Г.С.", "Капитан", "ЦВПС", "norma", "completed"),
    ("Уразов Б.Е.", "Майор", "1-я АА", "high", "completed"),
    ("Касымова М.Д.", "Полковник", "4-я БРХБЗ", "norma", "completed"),
]


def seed():
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            psych = User(login="psycholog", password_hash=hash_password("psycholog123"),
                         role="psychologist", full_name="Жумабеков А.С.",
                         rank="Подполковник", unit="Военный психолог")
            soldier = User(login="soldier", password_hash=hash_password("soldier123"),
                           role="soldier", full_name="Сейткали Н.А.",
                           rank="Майор", unit="7-я ОГБ · Алматы")
            db.add_all([psych, soldier])
            db.commit()
            db.add(Assignment(user_id=soldier.id, assigned_by="Подп. Жумабеков А.С.",
                              due="25 июня 2026", active=True))
            db.commit()
            print("Создано: 2 пользователя (psycholog/psycholog123, soldier/soldier123) + назначение")

        if db.query(Assessment).count() == 0:
            now = datetime.utcnow()
            for i, (name, rank, unit, label, status) in enumerate(DEMO):
                a = Assessment(
                    full_name=name, rank=rank, unit=unit, status=status, stage="done",
                    created_at=now - timedelta(hours=i),
                    completed_at=(now - timedelta(hours=i)) if status == "completed" else None,
                )
                if status == "completed":
                    sc, dsc = _scores_for(label)
                    a.scale_scores = sc
                    a.risk_levels_desc = dsc
                    a.npu = _npu_for(label)
                    a.risk_label = label
                    a.military_recommendation = {
                        "norma": "recommended", "medium": "recommended_with_restrictions",
                        "high": "not_recommended"}.get(label)
                    a.final_messages = ["Демо-запись для наполнения дашборда и списка."]
                db.add(a)
            db.commit()
            print(f"Создано демо-обследований: {len(DEMO)}")
        print("Сид завершён.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
