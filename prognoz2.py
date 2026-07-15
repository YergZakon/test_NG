"""Отдельный пайплайн «Прогноз-2»: опросник нервно-психической устойчивости.

Самодостаточный модуль — вся логика (вопросы, ключи, подсчёт) и UI (тест, результаты)
здесь. Основной адаптивный скрининг в app.py не затрагивается. Модуль использует
собственные ключи состояния (p2_responses / p2_current_index / p2_result), поэтому не
пересекается с состоянием основного потока.
"""

import io
import csv
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt

# --- Опросник «Прогноз-2»: вопросы, ключи и подсчёт результатов ---
# Номера пунктов 1-based, как в бланке методики.
PROGNOZ2_SINCERITY_NO = [1, 6, 10, 12, 15, 19, 21, 26, 33, 38, 44, 49, 52, 58, 61]
PROGNOZ2_NPU_YES = [2, 3, 5, 7, 9, 11, 13, 14, 16, 18, 20, 22, 23, 25, 27, 28, 29, 31, 32, 33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 48, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
PROGNOZ2_NPU_NO = [4, 8, 17, 24, 30, 35, 41, 46, 50, 55, 64]
PROGNOZ2_QUESTION_TEXTS = ["Бывало, что я бросал начатое дело, так как боялся, что не справлюсь с ним.", "Меня легко переспорить.", "Я избегаю поправлять людей, которые высказывают необоснованные утверждения.", "Люди проявляют ко мне столько сочувствия и симпатии, сколько я заслуживаю.", "Иногда я бываю уверен, что другие люди знают, о чем я думаю.", "Бывали случаи, что я не сдерживал своих обещаний.", "Временами я бываю совершенно уверен в своей никчемности.", "У меня никогда не было столкновений с законом.", "Я часто запоминаю числа, не имеющие для меня никакого значения (например, номера автомашин и т. п.).", "Иногда я говорю неправду.", "Я впечатлительнее большинства других людей.", "Мне приятно иметь среди своих знакомых значительных людей, это как бы придает мне вес в собственных глазах.", "Определенно судьба не благосклонна ко мне.", "Мне часто говорят, что я вспыльчив.", "Бывало, что я говорил о вещах, в которых не разбираюсь.", "Я легко теряю терпение с людьми.", "У меня отсутствуют враги, которые по-настоящему хотели бы причинить мне зло.", "Иногда мой слух настолько обостряется, что это мне даже мешает.", "Бывает, что я откладываю на завтра то, что можно сделать сегодня.", "Если бы люди не были настроены против меня, я достиг бы в жизни гораздо большего.", "В игре я предпочитаю выигрывать.", "Часто я перехожу на другую сторону улицы, чтобы избежать встречи с человеком, которого я не желаю видеть.", "Большую часть времени у меня такое чувство, будто я сделал что-то не то или даже плохое.", "Если кто-нибудь говорит глупость или как-нибудь иначе проявляет свое невежество, я стараюсь разъяснить ему его ошибку.", "Иногда у меня бывает чувство, что передо мной нагромоздилось столько трудностей, что одолеть их просто невозможно.", "В гостях я держусь за столом лучше, чем дома.", "В моей семье есть очень нервные люди.", "Если в моих неудачах кто-то виноват, я не оставляю его безнаказанным.", "Должен признать, что временами я волнуюсь из-за пустяков.", "Когда мне предлагают начать дискуссию или высказать мнение по вопросу, в котором я хорошо разбираюсь, я делаю это без робости.", "Я часто подшучиваю над друзьями.", "В течение жизни у меня несколько раз менялось отношение к моей профессии.", "Бывало, что при обсуждении некоторых вопросов я, особенно не задумываясь, соглашался с мнением других.", "Я часто работал под руководством людей, которые умели повернуть дело так, что все достижения в работе приписывались им, а виноватыми в ошибках оказывались другие.", "Я безо всякого страха вхожу в комнату, где другие уже собрались и разговаривают.", "Мне кажется, что по отношению именно ко мне особенно часто поступают несправедливо.", "Когда я нахожусь на высоком месте, у меня появляется желание прыгнуть вниз.", "Среди моих знакомых есть люди, которые мне не нравятся.", "Мои планы часто казались мне настолько трудно выполнимыми, что я должен был отказаться от них.", "Я часто бываю рассеянным и забывчивым.", "Приступы плохого настроения у меня бывают редко.", "Я бы предпочел работать с женщинами.", "Счастливей всего я бываю, когда я один.", "Иногда, когда я неважно себя чувствую, я бываю раздражительным.", "Часто я вижу сны, о которых лучше никому не рассказывать.", "Мои убеждения и взгляды непоколебимы.", "Я человек нервный и легковозбудимый.", "Меня очень раздражает, когда я забываю, куда кладу вещи.", "Бывает, что я сержусь.", "Работа, требующая пристального внимания, мне нравится.", "Иногда я бываю так взволнован, что не могу усидеть на месте.", "Бывает, что неприличная или даже непристойная шутка вызывает у меня смех.", "Иногда мне в голову приходят такие нехорошие мысли, что лучше о них никому не рассказывать.", "Иногда я принимаю валериану, элениум или другие успокаивающие средства.", "Человек я подвижный.", "Теперь мне трудно надеяться на то, что я чего-нибудь добьюсь в жизни.", "Иногда я чувствую, что близок к нервному срыву.", "Бывало, что я отвечал на письма не сразу после прочтения.", "Раз в неделю или чаще я бываю возбужденным и взволнованным.", "Мне очень трудно приспособиться к новым условиям жизни, работы или учебы. Переход к новым условиям жизни, работы или учебы кажется мне невыносимо трудным.", "Иногда случалось так, что я опаздывал на работу или свидание.", "Голова у меня болит часто.", "Я вел неправильный образ жизни.", "Алкогольные напитки я употребляю в умеренных количествах (или не употребляю вовсе).", "Я часто предаюсь грустным размышлениям.", "По сравнению с другими семьями в моей очень мало любви и тепла.", "У меня часто бывают подъемы и спады настроения.", "Когда я нахожусь среди людей, я слышу очень странные вещи.", "Я считаю, что меня очень часто наказывали незаслуженно.", "Мне страшно смотреть вниз с большой высоты.", "Бывало, что я целыми днями или даже неделями ничего не мог делать, потому что никак не мог заставить себя взяться за работу.", "Я ежедневно выпиваю необычно много воды.", "У меня бывали периоды, когда я что-то делал, а потом не знал, что именно я делал.", "Когда я пытаюсь что-то сделать, то часто замечаю, что у меня дрожат руки.", "Думаю, что я человек обреченный.", "У меня бывают периоды такого сильного беспокойства, что я даже не могу усидеть на месте.", "Временами мне кажется, что моя голова работает медленнее.", "Мне кажется, что я все чувствую более остро, чем другие.", "Иногда совершенно безо всякой причины у меня вдруг наступает период необычайной веселости.", "Некоторые вещи настолько меня волнуют, что мне даже говорить о них трудно.", "Иногда меня подводят нервы.", "Часто у меня бывает такое ощущение, будто все вокруг нереально.", "Когда я слышу об успехах близкого знакомого, я начинаю чувствовать, что я неудачник.", "Бывает, что мне в голову приходят плохие, часто даже ужасные слова, и я никак не могу от них отвязаться.", "Иногда я стараюсь держаться подальше от того или иного человека, чтобы не сделать или не сказать чего-нибудь такого, о чем потом пожалею.", "Часто, даже когда все складывается для меня хорошо, я чувствую, что мне все безразлично."]

PROGNOZ2_TOTAL = len(PROGNOZ2_QUESTION_TEXTS)  # 86
PROGNOZ2_MAX_NPU = len(PROGNOZ2_NPU_YES) + len(PROGNOZ2_NPU_NO)  # 72
PROGNOZ2_SINCERITY_THRESHOLD = 10  # >= 10 совпадений => недостоверно

PROGNOZ2_QUESTIONS = [
    {"id": f"p2_{index}", "number": index, "text": text, "scale": "prognoz2", "answer_type": "yes_no"}
    for index, text in enumerate(PROGNOZ2_QUESTION_TEXTS, start=1)
]


def prognoz2_raw_to_sten(raw_score):
    """Перевод первичного балла НПУ в стэны по ключу «Прогноз-2»."""
    if raw_score >= 43:
        return 1
    if raw_score >= 37:
        return 2
    if raw_score >= 33:
        return 3
    if raw_score >= 29:
        return 4
    if raw_score >= 23:
        return 5
    if raw_score >= 19:
        return 6
    if raw_score >= 15:
        return 7
    if raw_score >= 11:
        return 8
    if raw_score >= 9:
        return 9
    return 10


def prognoz2_interpret_sten(sten):
    """Интерпретация стэна: низкий стэн означает высокий риск срывов."""
    if sten in (1, 2, 3):
        return (
            "high",
            "высокий риск: низкий уровень нервно-психической устойчивости; рекомендуется очная оценка профильным специалистом",
        )
    if sten in (4, 5, 6):
        return (
            "medium",
            "средний риск: нервно-психическая устойчивость достаточная, но в напряженных ситуациях возможны срывы",
        )
    return (
        "low",
        "низкий риск: высокий уровень нервно-психической устойчивости, нервно-психические срывы маловероятны",
    )


def score_prognoz2(responses):
    """Подсчёт «Прогноз-2» по ключам методики.

    responses: dict {p2_id: bool} — ответы да/нет по вопросам.
    Возвращает dict с результатом. Бросает ValueError, если отвечены не все вопросы.
    """
    answers = {}
    for question in PROGNOZ2_QUESTIONS:
        value = responses.get(question["id"])
        if value is None:
            continue
        answers[question["number"]] = bool(value)

    missing = [number for number in range(1, PROGNOZ2_TOTAL + 1) if number not in answers]
    if missing:
        raise ValueError(f"Не отвечены вопросы Прогноз-2: {missing}")

    matched_sincerity = [
        number for number in PROGNOZ2_SINCERITY_NO
        if answers[number] is False
    ]
    matched_npu = [
        number
        for number in range(1, PROGNOZ2_TOTAL + 1)
        if (
            (number in PROGNOZ2_NPU_YES and answers[number] is True)
            or (number in PROGNOZ2_NPU_NO and answers[number] is False)
        )
    ]

    sincerity_score = len(matched_sincerity)
    raw_score = len(matched_npu)
    sten = prognoz2_raw_to_sten(raw_score)
    risk_level, conclusion = prognoz2_interpret_sten(sten)

    return {
        "sincerity_score": sincerity_score,
        "sincerity_valid": sincerity_score < PROGNOZ2_SINCERITY_THRESHOLD,
        "npu_raw_score": raw_score,
        "sten": sten,
        "risk_level": risk_level,
        "conclusion": conclusion,
        "matched_sincerity_items": matched_sincerity,
        "matched_npu_items": matched_npu,
    }


# --- UI пайплайна ---

def prepare_prognoz2():
    """Сброс состояния перед прохождением теста «Прогноз-2»."""
    st.session_state.p2_responses = {}
    st.session_state.p2_current_index = 0
    st.session_state.p2_result = None


def show_prognoz2_test():
    """Экран прохождения теста: один вопрос да/нет за раз."""
    st.title("📋 Опросник «Прогноз-2»")
    st.markdown(
        f"""
    Оценка нервно-психической устойчивости (В.Ю. Рыбников). Ответьте «Да» или «Нет»
    на {PROGNOZ2_TOTAL} утверждений.

    📝 **Помните**: нет правильных или неправильных ответов. Для корректного результата
    важны искренние ответы.
    """
    )

    total = len(PROGNOZ2_QUESTIONS)
    idx = st.session_state.p2_current_index
    question = PROGNOZ2_QUESTIONS[idx]
    progress = idx / total

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Вопрос {idx + 1} из {total}")
    with col2:
        st.metric("Прогресс", f"{int(progress * 100)}%")
    st.progress(progress)

    st.markdown(f"### 💭 {question['text']}")
    st.markdown("**Выберите ответ:**")

    selected_value = None
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Да", key=f"p2_btn_{question['id']}_yes", use_container_width=True):
            selected_value = True
    with col_no:
        if st.button("Нет", key=f"p2_btn_{question['id']}_no", use_container_width=True):
            selected_value = False

    if selected_value is not None:
        st.session_state.p2_responses[question["id"]] = selected_value

        if idx < total - 1:
            st.session_state.p2_current_index += 1
        else:
            # Все вопросы отвечены — считаем результат и переходим к результатам
            st.session_state.p2_result = score_prognoz2(st.session_state.p2_responses)
            st.session_state.stage = "p2_results"
        st.rerun()

    # Предыдущий ответ и навигация назад
    if question["id"] in st.session_state.p2_responses:
        prev = "Да" if st.session_state.p2_responses[question["id"]] else "Нет"
        st.info(f"Ваш предыдущий ответ: {prev}")

    if idx > 0:
        if st.button("⬅️ Предыдущий вопрос", key="p2_prev_btn"):
            st.session_state.p2_current_index -= 1
            st.rerun()


def _draw_sten_chart(result):
    """Компактная шкала стэна 1-10 с зонами риска."""
    color = {"high": "#F44336", "medium": "#FFC107", "low": "#4CAF50"}[result["risk_level"]]
    fig, ax = plt.subplots(figsize=(8, 1.7))
    ax.barh([0], [result["sten"]], color=color, height=0.5)
    ax.axvspan(0.5, 3.5, alpha=0.12, color="red")
    ax.axvspan(3.5, 6.5, alpha=0.12, color="gold")
    ax.axvspan(6.5, 10.5, alpha=0.12, color="green")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Стэн НПУ (1 — высокий риск, 10 — норма)")
    ax.set_title(f"Стэн: {result['sten']} из 10")
    plt.tight_layout()
    st.pyplot(fig)


def _prognoz2_report(result):
    """Текстовый отчёт для экспорта."""
    lines = [
        "ОТЧЕТ ПО ОПРОСНИКУ «ПРОГНОЗ-2»",
        "=" * 45,
        f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"ID сессии: {st.session_state.get('session_id', '')}",
        "",
        f"Сырой балл НПУ: {result['npu_raw_score']} из {PROGNOZ2_MAX_NPU}",
        f"Стэн: {result['sten']} из 10",
        f"Шкала искренности: {result['sincerity_score']} (порог {PROGNOZ2_SINCERITY_THRESHOLD})",
        "Достоверность: " + ("в норме" if result["sincerity_valid"] else "НИЗКАЯ — результаты под вопросом"),
        "",
        "ЗАКЛЮЧЕНИЕ:",
        result["conclusion"],
        "",
        "Психолог: ________________",
        f"Дата: {datetime.now().strftime('%d.%m.%Y')}",
    ]
    return "\n".join(lines)


def _prognoz2_csv(result):
    """CSV с результатом для экспорта."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Параметр", "Значение"])
    writer.writerow(["Сырой балл НПУ", result["npu_raw_score"]])
    writer.writerow(["Максимум НПУ", PROGNOZ2_MAX_NPU])
    writer.writerow(["Стэн", result["sten"]])
    writer.writerow(["Шкала искренности", result["sincerity_score"]])
    writer.writerow(["Порог искренности", PROGNOZ2_SINCERITY_THRESHOLD])
    writer.writerow(["Достоверность", "в норме" if result["sincerity_valid"] else "низкая"])
    writer.writerow(["Уровень риска", result["risk_level"]])
    writer.writerow(["Заключение", result["conclusion"]])
    return output.getvalue()


def show_prognoz2_results():
    """Экран результатов теста «Прогноз-2»."""
    st.title("📊 Результаты теста «Прогноз-2»")

    result = st.session_state.get("p2_result")
    if not result:
        st.warning("Нет данных о прохождении теста. Пожалуйста, пройдите тест.")
        if st.button("📋 Пройти тест «Прогноз-2»"):
            prepare_prognoz2()
            st.session_state.stage = "p2_test"
            st.rerun()
        return

    # Достоверность
    if not result["sincerity_valid"]:
        st.error(
            f"⚠️ **Низкая достоверность**: шкала искренности = {result['sincerity_score']} "
            f"(порог {PROGNOZ2_SINCERITY_THRESHOLD}). Результаты могут быть недостоверны."
        )

    # Метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Сырой балл НПУ", f"{result['npu_raw_score']} / {PROGNOZ2_MAX_NPU}")
    with col2:
        st.metric("Стэн", f"{result['sten']} / 10")
    with col3:
        st.metric("Достоверность", "в норме" if result["sincerity_valid"] else "под вопросом")

    # Заключение
    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[result["risk_level"]]
    if result["risk_level"] == "high":
        st.error(f"{emoji} {result['conclusion']}")
    elif result["risk_level"] == "medium":
        st.warning(f"{emoji} {result['conclusion']}")
    else:
        st.success(f"{emoji} {result['conclusion']}")

    # График стэна
    try:
        _draw_sten_chart(result)
    except Exception as exc:  # график не должен ломать экран результатов
        st.caption(f"(график недоступен: {exc})")

    st.markdown("---")

    # Экспорт
    st.subheader("💾 Экспорт результатов")
    session_id = st.session_state.get("session_id", "prognoz2")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Отчёт (TXT)",
            data=_prognoz2_report(result),
            file_name=f"prognoz2_{session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📊 Данные (CSV)",
            data=_prognoz2_csv(result),
            file_name=f"prognoz2_{session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # Управление
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Пройти заново", use_container_width=True):
            prepare_prognoz2()
            st.session_state.stage = "p2_test"
            st.rerun()
    with col2:
        if st.button("🏠 В начало", use_container_width=True):
            st.session_state.stage = "start"
            st.rerun()

    st.warning(
        """
    ### ⚕️ Важная информация

    Тест носит **предварительный** характер и не заменяет очной оценки специалистом.
    Окончательное решение принимает военно-врачебная комиссия.

    📞 **При кризисных ситуациях**: 8-800-2000-122 (психологическая помощь)
    """
    )
