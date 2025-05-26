import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import openai
import base64
import tempfile
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# Настройка страницы
st.set_page_config(
    page_title="Адаптивное психологическое тестирование",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определение глобальных переменных и констант
# Пороговые значения для шкал
THRESHOLDS = {
    "low": (3, 7),
    "medium": (8, 11),
    "high": (12, 15)
}

# Создание структуры данных для хранения вопросов

# Скрининговые вопросы (первичная оценка)
SCREENING_QUESTIONS = {
    "aggression": [
        {"id": "ag1", "text": "Иногда я не могу сдержать желание ударить другого человека."},
        {"id": "ag2", "text": "Я быстро вспыхиваю, но и быстро остываю."},
        {"id": "ag3", "text": "Я раздражаюсь, когда у меня что-то не получается."}
    ],
    "isolation": [
        {"id": "is1", "text": "Я чувствую себя совершенно одиноким."},
        {"id": "is2", "text": "Мне не хватает общения."},
        {"id": "is3", "text": "Я чувствую себя изолированным от других."}
    ],
    "somatic": [
        {"id": "som1", "text": "Иногда меня охватывает чувство ужаса"},
        {"id": "som2", "text": "Иногда я чувствую, что я не могу контролировать свои мысли"},
        {"id": "som3", "text": "Я не могу расслабиться"}
    ],
    "anxiety": [
        {"id": "anx1", "text": "Я испытываю напряженность, мне не по себе"},
        {"id": "anx2", "text": "У меня бывает внезапное чувство паники"},
        {"id": "anx3", "text": "Беспокойные мысли крутятся у меня в голове"}
    ],
    "stability": [
        {"id": "stab1", "text": "Временами я бываю совершенно уверен в своей никчемности."},
        {"id": "stab2", "text": "Часто я перехожу на другую сторону улицы, чтобы избежать встречи с человеком, которого я не желаю видеть."},
        {"id": "stab3", "text": "Иногда я чувствую, что близок к нервному срыву."}
    ],
    "sincerity": [
        {"id": "sin1", "text": "Бывало, что я говорил о вещах, в которых не разбираюсь."},
        {"id": "sin2", "text": "Бывает, что я сержусь."},
        {"id": "sin3", "text": "Иногда я говорю неправду."}
    ]
}

# Дополнительные вопросы для шкал со средним риском
MEDIUM_RISK_QUESTIONS = {
    "aggression": [
        {"id": "ag_med1", "text": "Я дерусь чаше, чем окружающие."},
        {"id": "ag_med2", "text": "Некоторые мои друзья считают, что я вспыльчив."},
        {"id": "ag_med3", "text": "Иногда я выхожу из себя без особой причины."},
        {"id": "ag_med4", "text": "Мне трудно сдерживать раздражение."},
        {"id": "ag_med5", "text": "Иногда я настолько выходил из себя, что ломал вещи."}
    ],
    "isolation": [
        {"id": "is_med1", "text": "Я несчастлив, занимаясь столькими вещами в одиночку."},
        {"id": "is_med2", "text": "Мне не с кем поговорить."},
        {"id": "is_med3", "text": "Я чувствую себя покинутым."},
        {"id": "is_med4", "text": "Я умираю по компании."},
        {"id": "is_med5", "text": "Я несчастен, будучи таким отверженным."},
        {"id": "is_med6", "text": "Мне трудно заводить друзей."}
    ],
    "somatic": [
        {"id": "som_med1", "text": "Иногда у меня бывает ускоренное сердцебиение"},
        {"id": "som_med2", "text": "Иногда я чувствую, что у меня удушье"},
        {"id": "som_med3", "text": "Иногда я чувствую, что у меня затрудненное дыхание"},
        {"id": "som_med4", "text": "Иногда я чувствую страх смерти"},
        {"id": "som_med5", "text": "Испуг"},
        {"id": "som_med6", "text": "Иногда у меня бывают желудочно-кишечные расстройства"}
    ],
    "anxiety": [
        {"id": "anx_med1", "text": "Я испытываю страх, кажется, будто что-то ужасное может вот-вот случиться"},
        {"id": "anx_med2", "text": "Я испытываю внутреннее напряжение или дрожь"},
        {"id": "anx_med3", "text": "Я испытываю неусидчивость, словно мне постоянно нужно двигаться"},
        {"id": "anx_med4", "text": "То, что приносило мне большое удовольствие, и сейчас вызывает у меня такое же чувство"},
        {"id": "anx_med5", "text": "Я могу получить удовольствие от хорошей книги, радио- или телепрограммы"}
    ],
    "stability": [
        {"id": "stab_med1", "text": "Определенно судьба не благосклонна ко мне."},
        {"id": "stab_med2", "text": "Я легко теряю терпение с людьми."},
        {"id": "stab_med3", "text": "Если бы люди не были настроены против меня, я достиг бы в жизни гораздо большего."},
        {"id": "stab_med4", "text": "Иногда у меня бывает чувство, что передо мной нагромоздилось столько трудностей, что одолеть их просто невозможно."},
        {"id": "stab_med5", "text": "Я часто работал под руководством людей, которые умели повернуть дело так, что все достижения в работе приписывались им, а виноватыми в ошибках оказывались другие."},
        {"id": "stab_med6", "text": "Я часто предаюсь грустным размышлениям."},
        {"id": "stab_med7", "text": "У меня часто бывают подъемы и спады настроения."}
    ]
}

# Полные опросники для шкал с высоким риском
# Для упрощения приложения включим только часть вопросов
HIGH_RISK_QUESTIONS = {
    "aggression": [
        {"id": "ag_full1", "text": "Иногда я не могу сдержать желание ударить другого человека."},
        {"id": "ag_full2", "text": "Я быстро вспыхиваю, но и быстро остываю."},
        {"id": "ag_full3", "text": "Бывает, что я просто схожу с ума от ревности."},
        {"id": "ag_full4", "text": "Если меня спровоцировать, я могу ударить другого человека."},
        {"id": "ag_full5", "text": "Я раздражаюсь, когда у меня что-то не получается."},
        {"id": "ag_full6", "text": "Временами мне кажется, что жизнь мне что-то недодала."},
        {"id": "ag_full7", "text": "Если кто-то ударит меня, я дам сдачи."},
        {"id": "ag_full8", "text": "Иногда я чувствую, что вот-вот взорвусь."},
        {"id": "ag_full9", "text": "Другим постоянно везет."},
        {"id": "ag_full10", "text": "Я дерусь чаше, чем окружающие."}
        # Остальные вопросы из полного опросника
    ],
    "isolation": [
        {"id": "is_full1", "text": "Я несчастлив, занимаясь столькими вещами в одиночку."},
        {"id": "is_full2", "text": "Мне не с кем поговорить."},
        {"id": "is_full3", "text": "Для меня невыносимо быть та­ким одиноким."},
        {"id": "is_full4", "text": "Мне не хватает общения."},
        {"id": "is_full5", "text": "Я чувствую, будто никто дей­ствительно не понимает себя."},
        {"id": "is_full6", "text": "Я застаю себя в ожидании, что люди позвонят или напишут мне."},
        {"id": "is_full7", "text": "Нет никого, к кому я мог бы обратиться."},
        {"id": "is_full8", "text": "Я сейчас больше ни с кем не близок."},
        {"id": "is_full9", "text": "Те, кто меня окружает, не раз­деляют мои интересы и идеи."},
        {"id": "is_full10", "text": "Я чувствую себя покинутым."}
        # Остальные вопросы из полного опросника
    ],
    "somatic": [
        {"id": "som_full1", "text": "Ощущение онемения или покалывания в теле"},
        {"id": "som_full2", "text": "Ощущение жары"},
        {"id": "som_full3", "text": "Дрожь в ногах"},
        {"id": "som_full4", "text": "Неспособность расслабиться"},
        {"id": "som_full5", "text": "Страх, что произойдет самое плохое"},
        {"id": "som_full6", "text": "Головокружение или ощущение легкости в голове"},
        {"id": "som_full7", "text": "Ускоренное сердцебиение"},
        {"id": "som_full8", "text": "Неустойчивость"},
        {"id": "som_full9", "text": "Ощущение ужаса"},
        {"id": "som_full10", "text": "Нервозность"}
        # Остальные вопросы из полного опросника
    ],
    "anxiety": [
        {"id": "anx_full1", "text": "Я испытываю напряженность, мне не по себе"},
        {"id": "anx_full2", "text": "То, что приносило мне большое удовольствие, и сейчас вызывает у меня такое же чувство"},
        {"id": "anx_full3", "text": "Я испытываю страх, кажется, будто что-то ужасное может вот-вот случиться"},
        {"id": "anx_full4", "text": "Я способен рассмеяться и увидеть в том или ином событии смешное"},
        {"id": "anx_full5", "text": "Беспокойные мысли крутятся у меня в голове"},
        {"id": "anx_full6", "text": "Я испытываю бодрость"},
        {"id": "anx_full7", "text": "Я легко могу сесть и расслабиться"},
        {"id": "anx_full8", "text": "Мне кажется, что я всё стал делать очень медленно"},
        {"id": "anx_full9", "text": "Я испытываю внутреннее напряжение или дрожь"},
        {"id": "anx_full10", "text": "Я не слежу за своей внешностью"}
        # Остальные вопросы из полного опросника
    ],
    "stability": [
        {"id": "stab_full1", "text": "Бывало, что я бросал начатое дело, так как боялся, что не справлюсь с ним."},
        {"id": "stab_full2", "text": "Меня легко переспорить."},
        {"id": "stab_full3", "text": "Я избегаю поправлять людей, которые высказывают необоснованные утверждения."},
        {"id": "stab_full4", "text": "Люди проявляют ко мне столько сочувствия и симпатии, сколько я заслуживаю."},
        {"id": "stab_full5", "text": "Иногда я бываю, уверен, что другие люди знают, о чем я думаю."},
        {"id": "stab_full6", "text": "Временами я бываю совершенно уверен в своей никчемности."},
        {"id": "stab_full7", "text": "Я часто запоминаю числа, не имеющие для меня никакого значения."},
        {"id": "stab_full8", "text": "Я впечатлительнее большинства других людей."},
        {"id": "stab_full9", "text": "Определенно судьба не благосклонна ко мне."},
        {"id": "stab_full10", "text": "Мне часто говорят, что я вспыльчив."}
        # Остальные вопросы из полного опросника
    ]
}

# Названия шкал
SCALE_NAMES = {
    "aggression": "Шкала агрессии (Басса-Перри)",
    "isolation": "Шкала изоляции/депривации (Д. Рассел)",
    "somatic": "Шкала соматической депрессии (Бека)",
    "anxiety": "Шкала тревожности и депрессии (NUDS)",
    "stability": "Шкала нервно-психической устойчивости",
    "sincerity": "Шкала искренности"
}

# Функция для установки API ключа OpenAI
def set_openai_api_key(api_key):
    """Устанавливает API ключ OpenAI как переменную окружения"""
    os.environ["OPENAI_API_KEY"] = api_key
    return True

# Функция для озвучивания текста с использованием OpenAI API
def generate_speech(text, voice="alloy", language="ru"):
    """
    Генерирует аудио для заданного текста с использованием OpenAI TTS API
    
    Args:
        text (str): Текст для озвучивания
        voice (str): Голос для использования (alloy, echo, fable, onyx, nova, shimmer)
        language (str): Код языка для озвучивания (en, ru, de, fr и т.д.)
    
    Returns:
        str: HTML код для проигрывания аудио в Streamlit
    """
    try:
        # Проверяем наличие API ключа
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.warning("API ключ OpenAI не настроен. Озвучивание недоступно.")
            return None
        
        # Инициализация клиента
        client = openai.OpenAI(api_key=api_key)
        
        # Создаем временный файл для сохранения аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_filename = temp_audio.name
        
        # Определяем инструкции для языка
        language_instructions = ""
        if language == "ru":
            language_instructions = "Говорите по-русски с четким произношением."
        elif language == "en":
            language_instructions = "Speak in English with clear pronunciation."
        elif language == "de":
            language_instructions = "Sprechen Sie Deutsch mit klarer Aussprache."
        elif language == "fr":
            language_instructions = "Parlez en français avec une prononciation claire."
        elif language == "es":
            language_instructions = "Hable en español con pronunciación clara."
        
        # Генерируем аудио
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="mp3",
            instructions=language_instructions
        )
        
        # Сохраняем аудио во временный файл
        response.stream_to_file(temp_filename)
        
        # Читаем аудио как base64 для встраивания в HTML
        with open(temp_filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Кодируем в base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Создаем HTML для audio player
        audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        
        # Удаляем временный файл
        os.unlink(temp_filename)
        
        return audio_html
    
    except Exception as e:
        st.error(f"Ошибка при генерации аудио: {str(e)}")
        return None

# Функции для работы с состоянием сессии
def initialize_session():
    """Инициализация состояния сессии при первом запуске"""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'start'
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'current_scale' not in st.session_state:
        st.session_state.current_scale = None
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'questions_order' not in st.session_state:
        st.session_state.questions_order = []
    if 'scale_scores' not in st.session_state:
        st.session_state.scale_scores = {}
    if 'risk_levels' not in st.session_state:
        st.session_state.risk_levels = {}
    if 'medium_risk_scales' not in st.session_state:
        st.session_state.medium_risk_scales = []
    if 'high_risk_scales' not in st.session_state:
        st.session_state.high_risk_scales = []
    if 'evaluated_scales' not in st.session_state:
        st.session_state.evaluated_scales = []
    if 'detailed_results' not in st.session_state:
        st.session_state.detailed_results = {}
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {'name': '', 'age': '', 'gender': ''}
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = False
    if 'tts_voice' not in st.session_state:
        st.session_state.tts_voice = 'alloy'
    if 'last_question_id' not in st.session_state:
        st.session_state.last_question_id = None
    if 'tts_language' not in st.session_state:
        st.session_state.tts_language = 'ru'

def reset_session():
    """Сброс состояния сессии для нового тестирования"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session()

def save_response(question_id, value):
    """Сохранение ответа на вопрос"""
    st.session_state.responses[question_id] = value

def prepare_screening_questions():
    """Подготовка порядка вопросов для первичного скрининга"""
    questions = []
    for scale, scale_questions in SCREENING_QUESTIONS.items():
        for question in scale_questions:
            questions.append({
                "id": question["id"],
                "text": question["text"],
                "scale": scale
            })
    # Перемешивание вопросов для более надежной оценки
    np.random.shuffle(questions)
    st.session_state.questions_order = questions
    st.session_state.current_question_index = 0

def prepare_detailed_questions(scale, risk_level):
    """Подготовка дополнительных вопросов для шкалы с повышенным риском"""
    if risk_level == "medium":
        questions = MEDIUM_RISK_QUESTIONS.get(scale, [])
    else:  # high risk
        questions = HIGH_RISK_QUESTIONS.get(scale, [])
    
    # Добавляем информацию о шкале к каждому вопросу
    for question in questions:
        question['scale'] = scale
    
    st.session_state.questions_order = questions
    st.session_state.current_question_index = 0
    st.session_state.current_scale = scale

def calculate_scale_score(scale, responses):
    """Расчет баллов по шкале на основе ответов"""
    score = 0
    count = 0
    
    # Учитываем ответы только на вопросы из указанной шкалы
    for question_id, value in responses.items():
        # Определяем, к какой шкале относится вопрос
        scale_key = None
        for s, questions in SCREENING_QUESTIONS.items():
            if any(q['id'] == question_id for q in questions):
                scale_key = s
                break
        
        if not scale_key and scale != 'sincerity':
            for s, questions in MEDIUM_RISK_QUESTIONS.items():
                if any(q['id'] == question_id for q in questions):
                    scale_key = s
                    break
            
        if not scale_key and scale != 'sincerity':
            for s, questions in HIGH_RISK_QUESTIONS.items():
                if any(q['id'] == question_id for q in questions):
                    scale_key = s
                    break
        
        if scale_key == scale:
            score += value
            count += 1
    
    return score, count

def analyze_screening_results():
    """Анализ результатов первичного скрининга"""
    scores = {}
    risk_levels = {}
    
    # Расчет баллов по каждой шкале
    for scale in SCREENING_QUESTIONS.keys():
        score, count = calculate_scale_score(scale, st.session_state.responses)
        if count > 0:  # предотвращает деление на ноль
            scores[scale] = score
            
            # Определение уровня риска
            if scale != 'sincerity':
                if THRESHOLDS["low"][0] <= score <= THRESHOLDS["low"][1]:
                    risk_levels[scale] = "low"
                elif THRESHOLDS["medium"][0] <= score <= THRESHOLDS["medium"][1]:
                    risk_levels[scale] = "medium"
                elif THRESHOLDS["high"][0] <= score <= THRESHOLDS["high"][1]:
                    risk_levels[scale] = "high"
            else:
                # Специальная логика для шкалы искренности
                if score >= 13:
                    risk_levels[scale] = "low_sincerity"
                elif score <= 4:
                    risk_levels[scale] = "low_sincerity"
                else:
                    risk_levels[scale] = "normal"
    
    st.session_state.scale_scores = scores
    st.session_state.risk_levels = risk_levels
    
    # Выделение шкал со средним и высоким риском
    medium_risk_scales = [s for s, level in risk_levels.items() if level == "medium" and s != 'sincerity']
    high_risk_scales = [s for s, level in risk_levels.items() if level == "high" and s != 'sincerity']
    
    st.session_state.medium_risk_scales = medium_risk_scales
    st.session_state.high_risk_scales = high_risk_scales
    
    # Определение следующего этапа тестирования
    if risk_levels.get('sincerity') == "low_sincerity":
        st.session_state.stage = 'sincerity_warning'
    elif high_risk_scales:
        st.session_state.stage = 'high_risk_assessment'
        # Выбираем первую шкалу с высоким риском для углубленной оценки
        st.session_state.current_scale = high_risk_scales[0]
        prepare_detailed_questions(high_risk_scales[0], "high")
    elif medium_risk_scales:
        st.session_state.stage = 'medium_risk_assessment'
        # Выбираем первую шкалу со средним риском для углубленной оценки
        st.session_state.current_scale = medium_risk_scales[0]
        prepare_detailed_questions(medium_risk_scales[0], "medium")
    else:
        st.session_state.stage = 'results'
        prepare_report()

def analyze_detailed_results(scale, risk_level):
    """Анализ результатов углубленной оценки"""
    # Сохраняем результаты углубленной оценки
    score, count = calculate_scale_score(scale, st.session_state.responses)
    
    # Вычисляем процент от максимального значения
    max_possible = count * 5  # 5 - максимальное значение для каждого вопроса
    percentage = (score / max_possible) * 100 if max_possible > 0 else 0
    
    result = {
        "score": score,
        "count": count,
        "percentage": percentage,
        "max_possible": max_possible
    }
    
    # Сохраняем детальные результаты
    st.session_state.detailed_results[scale] = result
    
    # Отмечаем шкалу как оцененную
    if scale not in st.session_state.evaluated_scales:
        st.session_state.evaluated_scales.append(scale)
    
    # Определяем следующий этап
    if risk_level == "medium":
        # Проверяем, нужно ли перейти к полной оценке для этой шкалы
        if percentage >= 60:  # Порог для перехода к полной оценке
            # Переходим к полной оценке этой шкалы
            prepare_detailed_questions(scale, "high")
            return
        
        # Удаляем шкалу из списка шкал со средним риском
        if scale in st.session_state.medium_risk_scales:
            st.session_state.medium_risk_scales.remove(scale)
        
        # Проверяем, есть ли еще шкалы со средним риском
        if st.session_state.medium_risk_scales:
            # Переходим к следующей шкале со средним риском
            next_scale = st.session_state.medium_risk_scales[0]
            st.session_state.current_scale = next_scale
            prepare_detailed_questions(next_scale, "medium")
        elif st.session_state.high_risk_scales:
            # Переходим к оценке шкал с высоким риском
            st.session_state.stage = 'high_risk_assessment'
            next_scale = st.session_state.high_risk_scales[0]
            st.session_state.current_scale = next_scale
            prepare_detailed_questions(next_scale, "high")
        else:
            # Все шкалы оценены, переходим к результатам
            st.session_state.stage = 'results'
            prepare_report()
    
    elif risk_level == "high":
        # Удаляем шкалу из списка шкал с высоким риском
        if scale in st.session_state.high_risk_scales:
            st.session_state.high_risk_scales.remove(scale)
        
        # Проверяем, есть ли еще шкалы с высоким риском
        if st.session_state.high_risk_scales:
            # Переходим к следующей шкале с высоким риском
            next_scale = st.session_state.high_risk_scales[0]
            st.session_state.current_scale = next_scale
            prepare_detailed_questions(next_scale, "high")
        else:
            # Проверяем, есть ли шкалы со средним риском, которые еще не оценены
            remaining_medium = [s for s in st.session_state.medium_risk_scales if s not in st.session_state.evaluated_scales]
            if remaining_medium:
                st.session_state.stage = 'medium_risk_assessment'
                next_scale = remaining_medium[0]
                st.session_state.current_scale = next_scale
                prepare_detailed_questions(next_scale, "medium")
            else:
                # Все шкалы оценены, переходим к результатам
                st.session_state.stage = 'results'
                prepare_report()

def prepare_report():
    """Подготовка отчета по результатам тестирования"""
    # Анализ результатов и формирование рекомендаций
    recommendations = []
    
    # Преобразуем баллы в уровни риска для отчета
    risk_levels_desc = {}
    for scale, score in st.session_state.scale_scores.items():
        if scale == 'sincerity':
            continue
            
        if score <= THRESHOLDS["low"][1]:
            risk_levels_desc[scale] = "низкий"
        elif score <= THRESHOLDS["medium"][1]:
            risk_levels_desc[scale] = "средний"
        else:
            risk_levels_desc[scale] = "высокий"
    
    # Дополняем описания на основе углубленной оценки
    for scale, result in st.session_state.detailed_results.items():
        percentage = result["percentage"]
        if percentage < 33:
            intensity = "низкой интенсивности"
        elif percentage < 67:
            intensity = "средней интенсивности"
        else:
            intensity = "высокой интенсивности"
        
        risk_levels_desc[scale] += f" ({intensity})"
    
    # Формируем рекомендации на основе уровней риска
    if any(level == "высокий" for level in risk_levels_desc.values()):
        recommendations.append("Рекомендуется консультация со специалистом для более детальной оценки выявленных рисков.")
    
    if "высокий" in risk_levels_desc.get("anxiety", ""):
        recommendations.append("Желательно освоение техник релаксации и управления стрессом для снижения тревожности.")
    
    if "высокий" in risk_levels_desc.get("somatic", ""):
        recommendations.append("Рекомендуется обратить внимание на физические проявления стресса и тревоги.")
    
    if "высокий" in risk_levels_desc.get("stability", ""):
        recommendations.append("Полезно развитие навыков эмоциональной саморегуляции и психологической устойчивости.")
    
    if "высокий" in risk_levels_desc.get("aggression", ""):
        recommendations.append("Рекомендуется обучение методам управления гневом и агрессией.")
    
    if "высокий" not in risk_levels_desc.get("isolation", ""):
        recommendations.append("Рекомендуется использовать имеющиеся социальные связи как ресурс для эмоциональной поддержки.")
    
    # Добавляем общие рекомендации
    recommendations.append("Регулярная физическая активность и здоровый сон значительно улучшают эмоциональное состояние.")
    
    # Сохраняем рекомендации в состоянии сессии
    st.session_state.recommendations = recommendations
    st.session_state.risk_levels_desc = risk_levels_desc

# Функции для отображения интерфейса

def show_start_screen():
    """Отображение начального экрана"""
    st.title("Адаптивное психологическое тестирование")
    
    st.markdown("""
    ## Добро пожаловать в систему адаптивного психологического тестирования
    
    Данная система разработана для оценки психологического состояния по нескольким ключевым шкалам:
    * Шкала агрессии (Басса-Перри)
    * Шкала изоляции/депривации (Д. Рассел)
    * Шкала соматической депрессии (Бека)
    * Шкала тревожности и депрессии (NUDS)
    * Шкала нервно-психической устойчивости
    
    **Преимущества адаптивного тестирования:**
    * Система задает дополнительные вопросы только по тем шкалам, где выявлены повышенные показатели
    * Это позволяет получить более точную оценку при меньшем количестве вопросов
    * Результаты тестирования включают персонализированные рекомендации
    
    **Инструкция:**
    1. Отвечайте на вопросы максимально искренне
    2. Для каждого утверждения выберите оценку от 1 до 5, где:
       * 1 - Совершенно не согласен
       * 2 - Скорее не согласен
       * 3 - Нейтрально
       * 4 - Скорее согласен
       * 5 - Полностью согласен
    3. Тестирование займет от 5 до 15 минут в зависимости от ваших ответов
    """)
    
    # Сбор информации о пользователе
    with st.form("user_info_form"):
        st.subheader("Для начала, пожалуйста, укажите информацию о себе")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Имя (опционально)", value=st.session_state.user_info.get('name', ''))
        with col2:
            age = st.number_input("Возраст", min_value=15, max_value=100, step=1, value=int(st.session_state.user_info.get('age', 25)) if st.session_state.user_info.get('age') else 25)
        
        # Устанавливаем пол только как "Мужской"
        gender = "Мужской"
        
        # Настройки озвучивания
        st.subheader("Настройки озвучивания вопросов")
        tts_enabled = st.checkbox("Включить озвучивание вопросов", value=st.session_state.tts_enabled)
        
        if tts_enabled:
            tts_voice = st.selectbox(
                "Выберите голос",
                ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                index=["alloy", "echo", "fable", "onyx", "nova", "shimmer"].index(st.session_state.tts_voice) if st.session_state.tts_voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] else 0
            )
            # Устанавливаем только русский язык
            tts_language = "ru"
        else:
            tts_voice = st.session_state.tts_voice
            tts_language = st.session_state.tts_language
        
        # Поле для ввода API ключа OpenAI для озвучивания
        api_key_expander = st.expander("Настройка озвучивания вопросов (API ключ OpenAI)")
        with api_key_expander:
            st.markdown("""
            Для озвучивания вопросов требуется API ключ OpenAI. Вы можете получить ключ API на сайте 
            [OpenAI](https://platform.openai.com/api-keys).
            """)
            
            api_key = st.text_input(
                "API ключ OpenAI",
                type="password",
                help="Введите ваш API ключ OpenAI для включения функции озвучивания"
            )
            
            if api_key:
                if st.button("Сохранить API ключ"):
                    if set_openai_api_key(api_key):
                        st.success("API ключ успешно сохранен! Функция озвучивания доступна.")
                    else:
                        st.error("Не удалось сохранить API ключ.")
        
        submitted = st.form_submit_button("Начать тестирование")
        
        if submitted:
            st.session_state.user_info = {
                'name': name,
                'age': age,
                'gender': gender
            }
            st.session_state.tts_enabled = tts_enabled
            st.session_state.tts_voice = tts_voice
            st.session_state.tts_language = tts_language
            st.session_state.stage = 'screening'
            prepare_screening_questions()
            st.rerun()
    
    # Кнопки для демонстрации
    with st.expander("Для разработчиков"):
        if st.button("Сбросить все и начать заново"):
            reset_session()
            st.rerun()
            
        # Информация о настройке OpenAI API ключа
        st.subheader("Настройка озвучивания вопросов")
        st.markdown("""
        Для работы функции озвучивания необходимо настроить API ключ OpenAI. 
        Вы можете получить ключ API на сайте [OpenAI](https://platform.openai.com/api-keys) и настроить его следующим образом:
        
        ```
        import os
        os.environ["OPENAI_API_KEY"] = "ваш-ключ-api"
        ```
        
        Или установить как переменную окружения перед запуском приложения:
        ```
        # Windows
        set OPENAI_API_KEY=ваш-ключ-api
        
        # Linux/MacOS
        export OPENAI_API_KEY=ваш-ключ-api
        ```
        """)

def show_question(question, progress=None):
    """Отображение вопроса с шкалой ответов"""
    st.subheader(f"Вопрос {st.session_state.current_question_index + 1} из {len(st.session_state.questions_order)}")
    
    if progress is not None:
        st.progress(progress)
    
    st.markdown(f"**{question['text']}**")
    
    # Добавляем озвучивание вопроса если оно включено
    if st.session_state.tts_enabled:
        audio_container = st.empty()
        
        # Проверяем, есть ли ключ OpenAI API
        if os.environ.get("OPENAI_API_KEY"):
            # Автоматическое озвучивание при появлении вопроса
            if 'last_question_id' not in st.session_state or st.session_state.last_question_id != question['id']:
                audio_html = generate_speech(question['text'], voice=st.session_state.tts_voice, language=st.session_state.tts_language)
                if audio_html:
                    audio_container.markdown(audio_html, unsafe_allow_html=True)
                    # Запоминаем ID вопроса, чтобы не озвучивать повторно при перерисовке страницы
                    st.session_state.last_question_id = question['id']
            
            # Кнопка для повторного озвучивания
            if st.button("🔊 Озвучить еще раз", key="tts_button"):
                audio_html = generate_speech(question['text'], voice=st.session_state.tts_voice, language=st.session_state.tts_language)
                if audio_html:
                    audio_container.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("Для озвучивания вопросов необходимо настроить API ключ OpenAI.")
    
    # Создаем радиокнопки для оценки от 1 до 5
    cols = st.columns(5)
    selected_value = None
    
    # Проверяем, был ли уже дан ответ на этот вопрос
    default_index = None
    if question['id'] in st.session_state.responses:
        default_index = st.session_state.responses[question['id']] - 1
    
    for i, col in enumerate(cols):
        value = i + 1
        label = f"{value}"
        with col:
            if col.button(label, key=f"btn_{value}", help=f"{value} - {'Совершенно не согласен' if value == 1 else 'Скорее не согласен' if value == 2 else 'Нейтрально' if value == 3 else 'Скорее согласен' if value == 4 else 'Полностью согласен'}"):
                selected_value = value
    
    # Если выбран ответ, переходим к следующему вопросу
    if selected_value is not None:
        save_response(question['id'], selected_value)
        
        # Проверяем, есть ли еще вопросы
        if st.session_state.current_question_index < len(st.session_state.questions_order) - 1:
            st.session_state.current_question_index += 1
        else:
            # Завершаем текущий этап тестирования
            if st.session_state.stage == 'screening':
                analyze_screening_results()
            elif st.session_state.stage == 'medium_risk_assessment':
                analyze_detailed_results(st.session_state.current_scale, "medium")
            elif st.session_state.stage == 'high_risk_assessment':
                analyze_detailed_results(st.session_state.current_scale, "high")
        
        st.rerun()
    
    # Описание шкалы
    st.markdown("""
    **Шкала оценки:**
    - **1** - Совершенно не согласен
    - **2** - Скорее не согласен
    - **3** - Нейтрально
    - **4** - Скорее согласен
    - **5** - Полностью согласен
    """)

def show_screening():
    """Отображение экрана первичного скрининга"""
    st.title("Первичный скрининг")
    
    st.markdown("""
    Пожалуйста, ответьте на следующие вопросы, выбирая оценку от 1 до 5.
    """)
    
    # Вычисляем прогресс
    progress = st.session_state.current_question_index / len(st.session_state.questions_order)
    
    # Отображаем текущий вопрос
    question = st.session_state.questions_order[st.session_state.current_question_index]
    show_question(question, progress)

def show_sincerity_warning():
    """Отображение предупреждения о возможной недостоверности ответов"""
    st.title("Внимание: Возможная недостоверность ответов")
    
    st.markdown("""
    ## Обратите внимание
    
    Система обнаружила возможную недостоверность в ваших ответах. Это может произойти по нескольким причинам:
    
    * Вы могли отвечать слишком однообразно
    * Возможно, вы не до конца поняли некоторые вопросы
    * Вы могли стремиться представить себя в определенном свете
    
    Для получения наиболее точных результатов и рекомендаций, важна максимальная искренность при ответах.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пройти скрининг заново"):
            reset_session()
            st.session_state.stage = 'screening'
            prepare_screening_questions()
            st.rerun()
    
    with col2:
        if st.button("Продолжить с имеющимися ответами"):
            # Продолжаем анализ, игнорируя проблему с искренностью
            st.session_state.risk_levels['sincerity'] = "warning_ignored"
            
            if st.session_state.high_risk_scales:
                st.session_state.stage = 'high_risk_assessment'
                st.session_state.current_scale = st.session_state.high_risk_scales[0]
                prepare_detailed_questions(st.session_state.high_risk_scales[0], "high")
            elif st.session_state.medium_risk_scales:
                st.session_state.stage = 'medium_risk_assessment'
                st.session_state.current_scale = st.session_state.medium_risk_scales[0]
                prepare_detailed_questions(st.session_state.medium_risk_scales[0], "medium")
            else:
                st.session_state.stage = 'results'
                prepare_report()
            
            st.rerun()

def show_detailed_assessment():
    """Отображение экрана углубленной оценки"""
    current_scale = st.session_state.current_scale
    scale_name = SCALE_NAMES.get(current_scale, current_scale)
    risk_level = "средним" if st.session_state.stage == 'medium_risk_assessment' else "высоким"
    
    st.title(f"Углубленная оценка: {scale_name}")
    
    st.markdown(f"""
    По результатам первичного скрининга выявлен {risk_level} риск по шкале "{scale_name}".
    
    Пожалуйста, ответьте на дополнительные вопросы для более точной оценки.
    """)
    
    # Вычисляем прогресс
    progress = st.session_state.current_question_index / len(st.session_state.questions_order)
    
    # Отображаем текущий вопрос
    question = st.session_state.questions_order[st.session_state.current_question_index]
    show_question(question, progress)

def show_results():
    """Отображение результатов тестирования"""
    st.title("Результаты психологического тестирования")
    
    # Добавление приветствия по имени, если оно указано
    if st.session_state.user_info.get('name'):
        st.markdown(f"## Здравствуйте, {st.session_state.user_info['name']}!")
    
    # Предупреждение о достоверности, если было
    if st.session_state.risk_levels.get('sincerity') == "warning_ignored":
        st.warning("⚠️ Обратите внимание: результаты могут иметь сниженную достоверность.")
    
    # Общая информация о тестировании
    st.markdown("""
    ### Информация о проведенном тестировании
    
    Вы прошли адаптивное психологическое тестирование, которое оценивает эмоциональное состояние по нескольким ключевым шкалам.
    
    Система адаптировалась к вашим ответам, задавая углубленные вопросы только по тем шкалам, где были выявлены повышенные показатели.
    """)
    
    # Отображение графика результатов
    st.subheader("Визуализация результатов по шкалам")
    
    # Подготовка данных для графика
    scales = []
    scores = []
    max_scores = []
    colors = []
    
    # Данные из первичного скрининга
    for scale, score in st.session_state.scale_scores.items():
        if scale == 'sincerity':
            continue
        
        scales.append(SCALE_NAMES.get(scale, scale))
        scores.append(score)
        # Максимум для первичного скрининга - 15 баллов (3 вопроса * 5 макс. баллов)
        max_scores.append(15)
        
        # Определение цвета в зависимости от уровня риска
        if score <= THRESHOLDS["low"][1]:
            colors.append('#4CAF50')  # Зеленый для низкого риска
        elif score <= THRESHOLDS["medium"][1]:
            colors.append('#FFC107')  # Желтый для среднего риска
        else:
            colors.append('#F44336')  # Красный для высокого риска
    
    # Данные из углубленной оценки
    for scale, result in st.session_state.detailed_results.items():
        scale_name = f"{SCALE_NAMES.get(scale, scale)} (углубл.)"
        
        scales.append(scale_name)
        scores.append(result['score'])
        max_scores.append(result['max_possible'])
        
        # Определение цвета в зависимости от процента
        percentage = result['percentage']
        if percentage < 33:
            colors.append('#4CAF50')  # Зеленый для низкого риска
        elif percentage < 67:
            colors.append('#FFC107')  # Желтый для среднего риска
        else:
            colors.append('#F44336')  # Красный для высокого риска
    
    # Создание DataFrame для графика
    df = pd.DataFrame({
        'Шкала': scales,
        'Балл': scores,
        'Максимум': max_scores,
        'Цвет': colors
    })
    
    # Вычисление процентов от максимума
    df['Процент'] = (df['Балл'] / df['Максимум']) * 100
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Шкала'], df['Процент'], color=df['Цвет'])
    
    # Добавление подписей значений
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{df["Балл"].iloc[i]}/{df["Максимум"].iloc[i]} ({width:.1f}%)',
                va='center')
    
    ax.set_xlabel('Процент от максимального значения')
    ax.set_title('Результаты психологического тестирования')
    ax.set_xlim(0, 105)  # Устанавливаем предел по оси X для размещения подписей
    
    # Добавление линий уровней риска
    ax.axvline(x=33, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=67, color='gray', linestyle='--', alpha=0.5)
    
    # Добавление подписей уровней риска
    ax.text(16.5, -0.5, 'Низкий риск', color='gray', ha='center')
    ax.text(50, -0.5, 'Средний риск', color='gray', ha='center')
    ax.text(83.5, -0.5, 'Высокий риск', color='gray', ha='center')
    
    st.pyplot(fig)
    
    # Подробное описание результатов
    st.subheader("Подробное описание результатов")
    
    for scale, level in st.session_state.risk_levels_desc.items():
        scale_name = SCALE_NAMES.get(scale, scale)
        
        if "высокий" in level:
            emoji = "🔴"
        elif "средний" in level:
            emoji = "🟡"
        else:
            emoji = "🟢"
        
        st.markdown(f"**{emoji} {scale_name}**: Уровень риска - {level}")
        
        # Добавление описаний для каждой шкалы
        if scale == "aggression":
            if "высокий" in level:
                st.markdown("Выявлен повышенный уровень агрессивных тенденций. Возможны трудности с контролем гнева и агрессивных импульсов.")
            elif "средний" in level:
                st.markdown("Умеренный уровень агрессивных тенденций. В стрессовых ситуациях может проявляться повышенная раздражительность.")
            else:
                st.markdown("Агрессивные тенденции в пределах нормы. Хороший контроль над эмоциями гнева.")
        
        elif scale == "isolation":
            if "высокий" in level:
                st.markdown("Выраженное чувство одиночества и социальной изоляции. Возможны трудности в установлении и поддержании социальных связей.")
            elif "средний" in level:
                st.markdown("Умеренное чувство одиночества. Периодически возникает ощущение недостатка общения и понимания со стороны окружающих.")
            else:
                st.markdown("Удовлетворительный уровень социальных связей. Чувство одиночества не выражено.")
        
        elif scale == "somatic":
            if "высокий" in level:
                st.markdown("Выраженные соматические проявления тревоги и депрессии. Возможны значительные физические дискомфортные ощущения, связанные с психологическим состоянием.")
            elif "средний" in level:
                st.markdown("Умеренные соматические проявления. Периодически возникают физические симптомы, связанные с тревогой или стрессом.")
            else:
                st.markdown("Незначительные соматические проявления. Хорошая устойчивость к физическим проявлениям стресса.")
        
        elif scale == "anxiety":
            if "высокий" in level:
                st.markdown("Высокий уровень тревожности. Присутствуют выраженное беспокойство, напряжение, возможны панические эпизоды.")
            elif "средний" in level:
                st.markdown("Умеренный уровень тревожности. Периодически возникает беспокойство и тревожные мысли, особенно в стрессовых ситуациях.")
            else:
                st.markdown("Низкий уровень тревожности. Хорошая эмоциональная стабильность.")
        
        elif scale == "stability":
            if "высокий" in level:
                st.markdown("Сниженная нервно-психическая устойчивость. Возможны трудности адаптации к стрессу, эмоциональная нестабильность.")
            elif "средний" in level:
                st.markdown("Умеренная нервно-психическая устойчивость. В сложных ситуациях могут возникать трудности с саморегуляцией.")
            else:
                st.markdown("Хорошая нервно-психическая устойчивость. Способность адекватно реагировать на стрессовые факторы.")
    
    # Рекомендации
    st.subheader("Рекомендации")
    
    for i, recommendation in enumerate(st.session_state.recommendations):
        st.markdown(f"{i+1}. {recommendation}")
    
    # Предупреждение о профессиональной консультации
    st.warning("""
    **Важно**: Данное тестирование не заменяет консультацию специалиста. 
    Если вас беспокоят какие-либо аспекты вашего психологического состояния, 
    рекомендуется обратиться к квалифицированному психологу или психотерапевту.
    """)
    
    # Кнопка для сохранения результатов
    if st.button("Сохранить результаты (PDF)"):
        # Здесь можно добавить код для создания PDF
        st.success("Эта функция будет доступна в следующей версии приложения.")
    
    # Кнопка для нового тестирования
    if st.button("Пройти новое тестирование"):
        reset_session()
        st.rerun()

# Основная функция приложения
def main():
    # Инициализация состояния сессии
    initialize_session()
    
    # Отображение соответствующего экрана в зависимости от текущего этапа
    if st.session_state.stage == 'start':
        show_start_screen()
    elif st.session_state.stage == 'screening':
        show_screening()
    elif st.session_state.stage == 'sincerity_warning':
        show_sincerity_warning()
    elif st.session_state.stage == 'medium_risk_assessment' or st.session_state.stage == 'high_risk_assessment':
        show_detailed_assessment()
    elif st.session_state.stage == 'results':
        show_results()
    else:
        st.error("Неизвестный этап тестирования. Пожалуйста, начните заново.")
        if st.button("Начать заново"):
            reset_session()
            st.rerun()

# Запуск приложения
if __name__ == "__main__":
    main()