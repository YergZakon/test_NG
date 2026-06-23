"""
Портал психологического тестирования ВС РК — FastAPI веб-приложение.
Реализует дизайн-образец (вариант A): Вход, Кабинет, Тест, Результаты, Дашборд, Список.
Переиспользует логику скоринга из scoring.py и хранит данные в Postgres/SQLite.
"""
import os
from datetime import datetime, date, timedelta

from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

import scoring
from database import engine, Base, get_db
from models import User, Assignment, Assessment
from auth import current_user

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Портал психологического тестирования ВС РК")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "dev-insecure-secret-change-me"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

Base.metadata.create_all(engine)

# ---- константы движка теста ----
SCALE_ORDER = list(scoring.SCREENING_QUESTIONS.keys())
SCREENING_FLAT = [(sc, q["id"], q["text"]) for sc in SCALE_ORDER for q in scoring.SCREENING_QUESTIONS[sc]]
STEP_LABELS = ["Военная анкета", "Псих. скрининг", "Углублённая", "Заключение"]
LIKERT = [
    (1, "Совсем нет · Мүлде жоқ"),
    (2, "Скорее нет · Сирек"),
    (3, "Иногда · Кейде"),
    (4, "Часто · Жиі"),
    (5, "Постоянно · Үнемі"),
]


def _user_ctx(u: User) -> dict:
    return {"full_name": u.full_name, "initials": u.initials, "rank": u.rank or "",
            "unit": u.unit or "", "role": u.role}


def _today_ru() -> str:
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]
    d = datetime.now()
    return f"{d.day} {months[d.month - 1]} {d.year}"


def _deep_questions(block: dict):
    bank = scoring.MEDIUM_RISK_QUESTIONS if block["level"] == "medium" else scoring.HIGH_RISK_QUESTIONS
    return bank.get(block["scale"], [])


def _compute_npu(risk_levels_desc: dict):
    high = sum(1 for d in risk_levels_desc.values() if "высокий" in d)
    med = sum(1 for d in risk_levels_desc.values() if "средний" in d)
    npu = max(1, min(10, 10 - 3 * high - 1 * med))
    label = "high" if high else ("medium" if med else "norma")
    return npu, label


def _npu_color(label: str) -> str:
    return {"high": "red", "medium": "orange", "norma": "green"}.get(label, "green")


# ============================================================
#  АВТОРИЗАЦИЯ
# ============================================================
@app.get("/", response_class=HTMLResponse)
def root(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u:
        return RedirectResponse("/login", status_code=303)
    return RedirectResponse("/dashboard" if u.role == "psychologist" else "/cabinet", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, db: Session = Depends(get_db)):
    if current_user(request, db):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {})


@app.post("/login", response_class=HTMLResponse)
def login_submit(request: Request, login: str = Form(...), password: str = Form(...),
                 db: Session = Depends(get_db)):
    from auth import verify_password
    user = db.query(User).filter(User.login == login.strip()).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(request, "login.html",
                                          {"error": "Неверный логин или пароль", "login": login})
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ============================================================
#  КАБИНЕТ ВОЕННОСЛУЖАЩЕГО
# ============================================================
@app.get("/cabinet", response_class=HTMLResponse)
def cabinet(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u:
        return RedirectResponse("/login", status_code=303)

    assignment = db.query(Assignment).filter(Assignment.user_id == u.id, Assignment.active == True).first()
    done = (db.query(Assessment)
            .filter(Assessment.user_id == u.id, Assessment.status == "completed")
            .order_by(Assessment.created_at.desc()).all())
    history = [{
        "id": a.id, "title": "Комплексное психологическое обследование",
        "date": (a.completed_at or a.created_at).strftime("%d.%m.%Y"),
        "npu": a.npu, "color": _npu_color(a.risk_label or "norma"),
        "verdict": _verdict_text(a.military_recommendation),
    } for a in done]

    return templates.TemplateResponse(request, "cabinet.html", {
        "user": _user_ctx(u), "active": "cabinet", "today": _today_ru(),
        "assignment": assignment, "history": history,
    })


def _verdict_text(rec: str) -> str:
    return {
        "recommended": "Годен к службе",
        "recommended_with_restrictions": "Годен с ограничениями",
        "not_recommended": "Не рекомендуется",
    }.get(rec, "—")


# ============================================================
#  ДВИЖОК ТЕСТА
# ============================================================
def _get_or_create_active(u: User, db: Session) -> Assessment:
    a = (db.query(Assessment)
         .filter(Assessment.user_id == u.id, Assessment.status == "in_progress")
         .order_by(Assessment.created_at.desc()).first())
    if not a:
        a = Assessment(user_id=u.id, status="in_progress", stage="anketa",
                       full_name=u.full_name, rank=u.rank, unit=u.unit,
                       questionnaire={}, responses={}, deep_plan=[], cursor={})
        db.add(a); db.commit(); db.refresh(a)
    return a


def _steps_ctx(stage: str):
    order = ["anketa", "screening", "deepening", "done"]
    cur = order.index(stage if stage in order else "anketa")
    out = []
    for i, lbl in enumerate(STEP_LABELS):
        status = "done" if i < cur else ("current" if i == cur else "wait")
        out.append({"num": i + 1, "label": lbl, "status": status})
    return out


@app.get("/test", response_class=HTMLResponse)
def test_get(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u or u.role != "soldier":
        return RedirectResponse("/login", status_code=303)
    a = _get_or_create_active(u, db)

    if a.stage == "anketa":
        sections = [{"title": scoring.QUESTIONNAIRE_FIELDS[s]["title"],
                     "fields": scoring.QUESTIONNAIRE_FIELDS[s]["fields"]} for s in scoring.SECTIONS]
        return templates.TemplateResponse(request, "anketa.html", {
            "user": _user_ctx(u), "steps": _steps_ctx("anketa"),
            "sections": sections, "values": a.questionnaire or {},
        })

    if a.stage == "screening":
        return _render_screening(request, u, a)

    if a.stage == "deepening":
        return _render_deepening(request, u, a)

    return RedirectResponse(f"/results/{a.id}", status_code=303)


def _render_screening(request, u, a):
    idx = (a.cursor or {}).get("screening_idx", 0)
    scale, qid, text = SCREENING_FLAT[idx]
    return templates.TemplateResponse(request, "test.html", {
        "user": _user_ctx(u), "steps": _steps_ctx("screening"),
        "progress": 10 + int(40 * (idx + 1) / len(SCREENING_FLAT)),
        "scale_label": scoring.SCALE_NAMES.get(scale, scale),
        "q_num": idx + 1, "q_total": len(SCREENING_FLAT),
        "question_text": text, "options": [{"value": v, "text": t} for v, t in LIKERT],
        "selected": (a.responses or {}).get(qid), "action": "/test",
        "can_back": idx > 0, "is_last": False,
    })


def _render_deepening(request, u, a):
    plan = a.deep_plan or []
    cur = a.cursor or {}
    bi, qi = cur.get("block_idx", 0), cur.get("q_idx", 0)
    block = plan[bi]
    qs = _deep_questions(block)
    q = qs[qi]
    total = sum(len(_deep_questions(b)) for b in plan)
    done_before = sum(len(_deep_questions(b)) for b in plan[:bi]) + qi
    return templates.TemplateResponse(request, "test.html", {
        "user": _user_ctx(u), "steps": _steps_ctx("deepening"),
        "progress": 50 + int(45 * (done_before + 1) / max(total, 1)),
        "scale_label": scoring.SCALE_NAMES.get(block["scale"], block["scale"]) + " · углублённо",
        "q_num": done_before + 1, "q_total": total,
        "question_text": q["text"], "options": [{"value": v, "text": t} for v, t in LIKERT],
        "selected": (a.responses or {}).get(q["id"]), "action": "/test",
        "can_back": False, "is_last": (bi == len(plan) - 1 and qi == len(qs) - 1),
    })


@app.post("/test/anketa")
async def test_anketa(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u or u.role != "soldier":
        return RedirectResponse("/login", status_code=303)
    a = _get_or_create_active(u, db)
    form = await request.form()
    data = {}
    for s in scoring.SECTIONS:
        for f in scoring.QUESTIONNAIRE_FIELDS[s]["fields"]:
            if f["type"] == "multiselect":
                data[f["id"]] = form.getlist(f["id"])
            else:
                data[f["id"]] = form.get(f["id"], "")
    a.questionnaire = data
    a.stage = "screening"
    a.cursor = {"screening_idx": 0}
    db.commit()
    return RedirectResponse("/test", status_code=303)


@app.post("/test")
async def test_answer(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u or u.role != "soldier":
        return RedirectResponse("/login", status_code=303)
    a = _get_or_create_active(u, db)
    form = await request.form()
    direction = form.get("dir", "next")
    answer = form.get("answer")

    if a.stage == "screening":
        idx = (a.cursor or {}).get("screening_idx", 0)
        if direction == "back":
            a.cursor = {"screening_idx": max(0, idx - 1)}
            db.commit(); return RedirectResponse("/test", status_code=303)
        if answer is None:
            return RedirectResponse("/test", status_code=303)
        scale, qid, _ = SCREENING_FLAT[idx]
        resp = dict(a.responses or {}); resp[qid] = int(answer); a.responses = resp
        if idx + 1 < len(SCREENING_FLAT):
            a.cursor = {"screening_idx": idx + 1}
            db.commit(); return RedirectResponse("/test", status_code=303)
        # скрининг завершён → анализ и план углубления
        res = scoring.analyze_screening(a.responses)
        a.scale_scores = res["scale_scores"]
        a.risk_levels = res["risk_levels"]
        a.risk_levels_desc = res["risk_levels_desc"]
        a.deep_plan = [{"scale": s, "level": "medium"} for s in res["medium_risk_scales"]]
        if a.deep_plan:
            a.stage = "deepening"; a.cursor = {"block_idx": 0, "q_idx": 0}
            db.commit(); return RedirectResponse("/test", status_code=303)
        return _finalize(a, db)

    if a.stage == "deepening":
        plan = a.deep_plan or []
        cur = a.cursor or {}
        bi, qi = cur.get("block_idx", 0), cur.get("q_idx", 0)
        block = plan[bi]; qs = _deep_questions(block)
        if answer is None:
            return RedirectResponse("/test", status_code=303)
        resp = dict(a.responses or {}); resp[qs[qi]["id"]] = int(answer); a.responses = resp
        if qi + 1 < len(qs):
            a.cursor = {"block_idx": bi, "q_idx": qi + 1}
            db.commit(); return RedirectResponse("/test", status_code=303)
        # блок шкалы завершён → оценка и возможная эскалация
        ev = scoring.evaluate_detailed(block["scale"], block["level"], a.responses)
        det = dict(a.detailed_results or {}); det[block["scale"]] = ev; a.detailed_results = det
        rld = dict(a.risk_levels_desc or {}); rld[block["scale"]] = ev["new_risk_desc"]; a.risk_levels_desc = rld
        new_plan = list(plan)
        if block["level"] == "medium" and ev.get("escalate_to_high"):
            new_plan.insert(bi + 1, {"scale": block["scale"], "level": "high"})
        a.deep_plan = new_plan
        if bi + 1 < len(new_plan):
            a.cursor = {"block_idx": bi + 1, "q_idx": 0}
            db.commit(); return RedirectResponse("/test", status_code=303)
        return _finalize(a, db)

    return RedirectResponse(f"/results/{a.id}", status_code=303)


def _finalize(a: Assessment, db: Session):
    rec = scoring.final_recommendation(a.questionnaire or {}, a.risk_levels_desc or {})
    a.military_recommendation = rec["recommendation"]
    a.final_messages = rec["messages"]
    npu, label = _compute_npu(a.risk_levels_desc or {})
    a.npu = npu; a.risk_label = label
    a.stage = "done"; a.status = "completed"; a.completed_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(f"/results/{a.id}", status_code=303)


# ============================================================
#  РЕЗУЛЬТАТЫ
# ============================================================
def _scales_view(a: Assessment):
    rows = []
    for scale, name in scoring.SCALE_NAMES.items():
        score = (a.scale_scores or {}).get(scale)
        if score is None:
            continue
        maxv = len(scoring.SCREENING_QUESTIONS[scale]) * 5
        desc = (a.risk_levels_desc or {}).get(scale, "")
        if scale == "sincerity":
            badge, bcolor, fill = ("Достоверно", "blue", "navy")
            if "низкая" in desc:
                badge, bcolor = ("Сомнительно", "orange")
        elif "высокий" in desc:
            badge, bcolor, fill = ("Высокий", "red", "red")
        elif "средний" in desc:
            badge, bcolor, fill = ("Средний", "orange", "orange")
        else:
            badge, bcolor, fill = ("Норма", "green", "green")
        rows.append({"name": name, "score": score, "max": maxv,
                     "badge": badge, "badge_color": bcolor, "fill": fill,
                     "pct": round(score / maxv * 100) if maxv else 0})
    return rows


@app.get("/results", response_class=HTMLResponse)
def results_latest(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u:
        return RedirectResponse("/login", status_code=303)
    a = (db.query(Assessment)
         .filter(Assessment.user_id == u.id, Assessment.status == "completed")
         .order_by(Assessment.created_at.desc()).first())
    if not a:
        return RedirectResponse("/cabinet", status_code=303)
    return RedirectResponse(f"/results/{a.id}", status_code=303)


@app.get("/results/{aid}", response_class=HTMLResponse)
def results_view(aid: int, request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u:
        return RedirectResponse("/login", status_code=303)
    a = db.get(Assessment, aid)
    if not a or (u.role == "soldier" and a.user_id != u.id):
        return RedirectResponse("/", status_code=303)

    npu_cap = {"high": "Низкая НПУ", "medium": "Удовл. НПУ", "norma": "Хорошая НПУ"}.get(a.risk_label, "—")
    return templates.TemplateResponse(request, "results.html", {
        "user": _user_ctx(u), "active": "results", "a": a,
        "scales": _scales_view(a),
        "verdict": _verdict_text(a.military_recommendation),
        "verdict_color": _npu_color(a.risk_label or "norma"),
        "npu_color": _npu_color(a.risk_label or "norma"),
        "npu_cap": npu_cap,
        "date": (a.completed_at or a.created_at).strftime("%d.%m.%Y"),
    })


@app.get("/results/{aid}/report.txt")
def results_report(aid: int, request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u:
        return RedirectResponse("/login", status_code=303)
    a = db.get(Assessment, aid)
    if not a or (u.role == "soldier" and a.user_id != u.id):
        return RedirectResponse("/", status_code=303)
    lines = [
        "ЗАКЛЮЧЕНИЕ ПСИХОЛОГИЧЕСКОГО ОБСЛЕДОВАНИЯ — ВС РК",
        "=" * 56,
        f"ФИО: {a.full_name}", f"Звание: {a.rank or '—'} · Подразделение: {a.unit or '—'}",
        f"Дата: {(a.completed_at or a.created_at).strftime('%d.%m.%Y')}",
        f"НПУ: {a.npu}/10 · Заключение: {_verdict_text(a.military_recommendation)}",
        "", "Психологические шкалы:",
    ]
    for r in _scales_view(a):
        lines.append(f"  - {r['name']}: {r['score']}/{r['max']} ({r['badge']})")
    lines += ["", "Рекомендации:"] + [f"  {m}" for m in (a.final_messages or [])]
    return PlainTextResponse("\n".join(lines), headers={
        "Content-Disposition": f'attachment; filename="report_{a.id}.txt"'})


# ============================================================
#  ПСИХОЛОГ: ДАШБОРД и СПИСОК
# ============================================================
def _risk_badge(label: str):
    return {"high": ("Высокий", "red"), "medium": ("Средний", "orange"),
            "norma": ("Норма", "green")}.get(label, ("В процессе", "grey"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u or u.role != "psychologist":
        return RedirectResponse("/login", status_code=303)

    now = datetime.utcnow()
    start_day = datetime(now.year, now.month, now.day)
    start_week = start_day - timedelta(days=now.weekday())
    completed = db.query(Assessment).filter(Assessment.status == "completed").all()

    today_n = sum(1 for a in completed if (a.completed_at or a.created_at) >= start_day)
    week_n = sum(1 for a in completed if (a.completed_at or a.created_at) >= start_week)
    high_n = sum(1 for a in completed if a.risk_label == "high")
    norma_n = sum(1 for a in completed if a.risk_label == "norma")

    recent_rows = (db.query(Assessment).order_by(Assessment.created_at.desc()).limit(6).all())
    recent = []
    for a in recent_rows:
        badge, color = _risk_badge(a.risk_label if a.status == "completed" else None)
        recent.append({
            "initials": _initials(a.full_name), "name": a.full_name,
            "sub": f"{a.rank or ''} · {a.unit or ''}".strip(" ·"),
            "tag": (f"НПУ {a.npu}" if a.status == "completed" else "В процессе"),
            "color": color, "time": (a.completed_at or a.created_at).strftime("%H:%M"),
        })

    buckets = [("НПУ 9–10", "green", range(9, 11)), ("НПУ 7–8", "green", range(7, 9)),
               ("НПУ 5–6", "orange", range(5, 7)), ("НПУ 3–4", "red", range(3, 5)),
               ("НПУ 1–2", "red", range(1, 3))]
    dist = []
    npus = [a.npu for a in completed if a.npu is not None]
    maxc = max([sum(1 for n in npus if n in rng) for _, _, rng in buckets] + [1])
    for lbl, color, rng in buckets:
        c = sum(1 for n in npus if n in rng)
        dist.append({"label": lbl, "color": color, "count": c, "pct": round(c / maxc * 100)})

    return templates.TemplateResponse(request, "dashboard.html", {
        "user": _user_ctx(u), "active": "dashboard", "today": _today_ru(),
        "stats": {"today": today_n, "week": week_n, "high": high_n, "norma": norma_n},
        "recent": recent, "dist": dist,
    })


@app.get("/list", response_class=HTMLResponse)
def personnel_list(request: Request, q: str = "", db: Session = Depends(get_db)):
    u = current_user(request, db)
    if not u or u.role != "psychologist":
        return RedirectResponse("/login", status_code=303)
    query = db.query(Assessment).order_by(Assessment.created_at.desc())
    rows_all = query.all()
    if q:
        ql = q.lower()
        rows_all = [a for a in rows_all if ql in (a.full_name or "").lower()
                    or ql in (a.rank or "").lower() or ql in (a.unit or "").lower()]
    rows = []
    for i, a in enumerate(rows_all, 1):
        badge, color = _risk_badge(a.risk_label if a.status == "completed" else None)
        rows.append({
            "n": i, "id": a.id, "name": a.full_name, "rank": a.rank or "—", "unit": a.unit or "—",
            "date": (a.completed_at or a.created_at).strftime("%d.%m"),
            "npu": a.npu if a.status == "completed" else "—",
            "npu_color": _npu_color(a.risk_label or "norma") if a.status == "completed" else "navy",
            "badge": badge, "badge_color": color,
        })
    return templates.TemplateResponse(request, "list.html", {
        "user": _user_ctx(u), "active": "list", "rows": rows, "q": q, "total": len(rows),
    })


def _initials(name: str) -> str:
    parts = (name or "").split()
    if not parts:
        return "—"
    s = parts[0][:1] + (parts[1][:1] if len(parts) > 1 else "")
    return s.upper()
