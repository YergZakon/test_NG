import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import json
import os
import base64
import tempfile

# Попытка импорта OpenAI для аудио (только для локального запуска)
try:
    import openai
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.sidebar.warning("🔊 OpenAI не установлен. Аудио функции недоступны.")

# Настройка страницы
st.set_page_config(
    page_title="Система психологического тестирования военнослужащих",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определение глобальных переменных и констант
THRESHOLDS = {
    "low": (3, 7),
    "medium": (8, 11),
    "high": (12, 15)
}

# Скрининговые вопросы (первичная оценка)
SCREENING_QUESTIONS = {
    "aggression": [
        {"id": "ag1", "text": "Я раздражаюсь, когда у меня что-то не получается."},
        {"id": "ag2", "text": "Иногда, когда я неважно себя чувствую, я бываю раздражительным."},
        {"id": "ag3", "text": "Некоторые мои друзья считают, что я вспыльчив."}
    ],
    "isolation": [
        {"id": "is1", "text": "Мне трудно заводить друзей."},
        {"id": "is2", "text": "Мне не хватает общения."},
        {"id": "is3", "text": "Мне не с кем поговорить."}
    ],
    "somatic": [
        {"id": "som1", "text": "Иногда у меня бывает ускоренное сердцебиение"},
        {"id": "som2", "text": "Иногда я чувствую, что я не могу контролировать свои мысли"},
        {"id": "som3", "text": "Иногда у меня бывают желудочно-кишечные расстройства "}
    ],
    "anxiety": [
        {"id": "anx1", "text": "Я испытываю напряженность, мне не по себе"},
        {"id": "anx2", "text": "Приступы плохого настроения у меня бывают редко."},
        {"id": "anx3", "text": "Иногда совершенно безо всякой причины у меня вдруг наступает период необычайной веселости."}
    ],
    "stability": [
        {"id": "stab1", "text": "Я могу получить удовольствие от хорошей книги, радио- или телепрограммы "},
        {"id": "stab2", "text": "Бывало, что при обсуждении некоторых вопросов я, особенно не задумываясь, соглашался с мнением других."},
        {"id": "stab3", "text": "У меня часто бывают подъемы и спады настроения."}
    ],
    "military_adaptation": [
        {"id": "mil1", "text": "Мне трудно выполнять приказы без объяснения причин."},
        {"id": "mil2", "text": "Я боюсь физических нагрузок и испытаний."},
        {"id": "mil3", "text": "Мне сложно находиться далеко от дома длительное время."}
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
        {"id": "ag_med1", "text": "Я дерусь чаще, чем окружающие."},
        {"id": "ag_med2", "text": "Если кто-то ударит меня, я дам сдачи."},
        {"id": "ag_med3", "text": "Иногда я выхожу из себя без особой причины."},
        {"id": "ag_med4", "text": "Мне трудно сдерживать раздражение."},
        {"id": "ag_med5", "text": "Иногда я настолько выходил из себя, что ломал вещи."}
    ],
    "isolation": [
        {"id": "is_med1", "text": "Счастливей всего я бываю, когда я один."},
        {"id": "is_med2", "text": "Если бы люди не были настроены против меня, я достиг бы в жизни гораздо большего."},
        {"id": "is_med3", "text": "Иногда я бываю, уверен, что другие люди знают, о чем я думаю."},
        {"id": "is_med4", "text": "Мне кажется, что по отношению именно ко мне особенно часто поступают несправедливо."},
        {"id": "is_med5", "text": "Часто, даже когда все складывается для меня хорошо, я чувствую, что мне все безразлично."},
        {"id": "is_med6", "text": "Мне кажется, что я все чувствую более остро, чем другие."}
    
    ],
    "somatic": [
        {"id": "som_med1", "text": "Бывало, что я целыми днями или даже неделями ничего не мог делать, потому что никак не мог заставить себя взяться за работу."},
        {"id": "som_med2", "text": "Иногда я чувствую, что у меня удушье"},
        {"id": "som_med3", "text": "Иногда я чувствую, что у меня затрудненное дыхание"},
        {"id": "som_med4", "text": "Когда я пытаюсь что-то сделать, то часто замечаю, что у меня дрожат руки."},
        {"id": "som_med5", "text": "Иногда я чувствую Испуг"},
        {"id": "som_med6", "text": "Беспокойные мысли крутятся у меня в голове"}
    ],
    "anxiety": [
        {"id": "anx_med1", "text": "у меня бывает внезапное чуство паники"},
        {"id": "anx_med2", "text": "Я испытываю внутреннее напряжение или дрожь"},
        {"id": "anx_med3", "text": "Я испытываю неусидчивость, словно мне постоянно нужно двигаться"},
        {"id": "anx_med4", "text": "То, что приносило мне большое удовольствие, и сейчас вызывает у меня такое же чувство"},
        {"id": "anx_med5", "text": "Работа, требующая пристального внимания, мне нравится."}
    ],
    "stability": [
        {"id": "stab_med1", "text": "Определенно судьба не благосклонна ко мне."},
        {"id": "stab_med2", "text": "Я легко теряю терпение с людьми."},
        {"id": "stab_med3", "text": "Люди проявляют ко мне столько сочувствия и симпатии, сколько я заслуживаю."},
        {"id": "stab_med4", "text": "Иногда мне в голову приходят такие нехорошие мысли, что лучше о них никому не рассказывать."},
        {"id": "stab_med5", "text": "Должен признать, что временами я волнуюсь из-за пустяков."},
        {"id": "stab_med6", "text": "Я часто предаюсь грустным размышлениям."},
        {"id": "stab_med7", "text": "Я человек нервный и легковозбудимый.."}
    ],
    "military_adaptation": [
        {"id": "mil_med1", "text": "Мне трудно принимать решения в стрессовых ситуациях."},
        {"id": "mil_med2", "text": "Я плохо переношу критику от старших."},
        {"id": "mil_med3", "text": "Мне сложно работать в команде."},
        {"id": "mil_med4", "text": "Я избегаю ответственности за других людей."},
        {"id": "mil_med5", "text": "Мне трудно соблюдать строгий распорядок дня."}
    ]
}

# Полные опросники для шкал с высоким риском
HIGH_RISK_QUESTIONS = {
    "aggression": [
        {"id": "ag_full1", "text": "Иногда я не могу сдержать желание ударить другого человека."},
        {"id": "ag_full2", "text": "Я быстро вспыхиваю, но и быстро остываю."},
        {"id": "ag_full3", "text": "Бывает, что я просто схожу с ума от ревности."},
        {"id": "ag_full4", "text": "Если меня спровоцировать, я могу ударить другого человека."},
        {"id": "ag_full5", "text": "Иногда я не могу сдержать желание ударить другого человека."},
        {"id": "ag_full6", "text": "Временами мне кажется, что жизнь мне что-то недодала."},
        {"id": "ag_full7", "text": "Я легко теряю терпение с людьми."},
        {"id": "ag_full8", "text": "Иногда я чувствую, что вот-вот взорвусь."},
        {"id": "ag_full9", "text": "Другим постоянно везет."},
        {"id": "ag_full10", "text": "Я дерусь чаще, чем окружающие."}
    ],
    "isolation": [
        {"id": "is_high1", "text": "Я несчастлив, занимаясь столькими вещами в одиночку."},
        {"id": "is_high2", "text": "Я чувствую себя изолированным от других."},
        {"id": "is_high3", "text": "Я чувствую себя покинутым."},
        {"id": "is_high4", "text": "Я впечатлительнее большинства других людей."},
        {"id": "is_high5", "text": "я несчастен будучи таким отверженным."},
        {"id": "is_high6", "text": "Я чувствую себя совершенно одиноким."}
     ],
    "somatic": [
        {"id": "som_high1", "text": "Голова у меня болит часто."},
        {"id": "som_high2", "text": "Иногда мой слух настолько обостряется, что это мне даже мешает."},
        {"id": "som_high3", "text": "Иногда я чувствую, что у меня затрудненное дыхание"},
        {"id": "som_high4", "text": "Иногда я чувствую страх смерти"},
        {"id": "som_high5", "text": "Раз в неделю или чаще я бываю возбужденным и взволнованным."},
        {"id": "som_high6", "text": "Иногда я принимаю валериану, элениум или другие успокаивающие средства."}
    ],
    "anxiety": [
        {"id": "anx_high1", "text": "Я испытываю страх, кажется, будто что-то ужасное может вот-вот случиться"},
        {"id": "anx_high2", "text": "Некоторые вещи настолько меня волнуют, что мне даже говорить о них трудно."},
        {"id": "anx_high3", "text": "Иногда меня подводят нервы"},
        {"id": "anx_high4", "text": "Думаю, что я человек обреченный."},
        {"id": "anx_high5", "text": "Временами я бываю совершенно уверен в своей никчемности."}
    ],
    "stability": [
        {"id": "stab_high1", "text": "Теперь мне трудно надеяться на то, что я чего-нибудь добьюсь в жизни."},
        {"id": "stab_high2", "text": "Я легко теряю терпение с людьми."},
        {"id": "stab_high3", "text": "У меня бывали периоды, когда я что-то делал, а потом не знал, что именно я делал."},
        {"id": "stab_high4", "text": "Иногда у меня бывает чувство, что передо мной нагромоздилось столько трудностей, что одолеть их просто невозможно."},
        {"id": "stab_high5", "text": "Если в моих неудачах кто-то виноват, я не оставляю его безнаказанным."},
        {"id": "stab_high6", "text": "Мне очень трудно приспособиться к новым условиям жизни, работы или учебы. Переход к новым условиям жизни, работы или учебы кажется мне невыносимо трудным."},
        {"id": "stab_high7", "text": "Иногда я чувствую, что близок к нервному срыву."}
    ],
    "military_adaptation": [
        {"id": "mil_high1", "text": "Мне трудно принимать решения в стрессовых ситуациях."},
        {"id": "mil_high2", "text": "Я плохо переношу критику от старших."},
        {"id": "mil_high3", "text": "Мне сложно работать в команде."},
        {"id": "mil_high4", "text": "Я избегаю ответственности за других людей."},
        {"id": "mil_high5", "text": "Мне трудно соблюдать строгий распорядок дня."}
    ]
}
   

# Названия шкал
SCALE_NAMES = {
    "aggression": "Шкала агрессии (Басса-Перри)",
    "isolation": "Шкала изоляции/депривации (Д. Рассел)",
    "somatic": "Шкала соматической депрессии (Бека)",
    "anxiety": "Шкала тревожности и депрессии (NUDS)",
    "stability": "Шкала нервно-психической устойчивости",
    "military_adaptation": "Шкала военной адаптации",
    "sincerity": "Шкала искренности"
}

# Расширенная анкета для военнослужащих
MILITARY_QUESTIONNAIRE = {
    "personal_info": {
        "title": "👤 Личная информация",
        "questions": [
            {"id": "full_name", "text": "ФИО", "type": "text", "required": True},
            {"id": "birth_date", "text": "Дата рождения", "type": "date", "required": True},
            {"id": "birth_place", "text": "Место рождения", "type": "text", "required": True},
            {"id": "residence", "text": "Место жительства", "type": "text", "required": True},   
            {"id": "residence_coliving", "text": "С кем в настоящее время проживаете и в течении какого времени", "type": "text", "required": True},
            {"id": "team_senior", "text": "Старший команды", "type": "text", "required": False},
            {"id": "nationality", "text": "Национальность", "type": "text", "required": True},
            {"id": "marital_status", "text": "Семейное положение", "type": "select", "options": ["Холост", "Женат", "Разведен"], "required": True},
            {"id": "education", "text": "Образование", "type": "select", "options": ["Среднее", "Среднее специальное", "Высшее", "Неполное высшее"], "required": True},
            {"id": "social_media", "text": "Укажите ваши аккаунты в соц сетях", "type": "textarea", "required": False}
        ]
    },
    "achievements_family": {
        "title": "🏆 Достижения и семья",
        "questions": [
            {"id": "sports_achievements", "text": "Есть ли у вас спортивные достижения? Какие?", "type": "textarea", "required": False},
            {"id": "family_completeness", "text": "Вы воспитывались в полной/неполной семье", "type": "select", "options": ["Полной", "Неполной"], "required": True},
            {"id": "deceased_relatives", "text": "Есть ли умершие среди близких родственников? (кто, год смерти, причина)", "type": "textarea", "required": False}
        ]
    },
    "family_info": {
        "title": "👨‍👩‍👧‍👦 Информация о семье",
        "questions": [
            {"id": "father_info", "text": "ФИО отца, возраст, место работы", "type": "textarea", "required": False},
            {"id": "father_relationship", "text": "Взаимоотношения с отцом", "type": "select", "options": ["Отличные", "Хорошие", "Удовлетворительные", "Плохие", "Отсутствуют"], "required": False},
            {"id": "mother_info", "text": "ФИО матери, возраст, место работы", "type": "textarea", "required": False},
            {"id": "mother_relationship", "text": "Взаимоотношения с матерью", "type": "select", "options": ["Отличные", "Хорошие", "Удовлетворительные", "Плохие", "Отсутствуют"], "required": False},
            {"id": "siblings", "text": "Братья и сестры (ФИО, возраст)", "type": "textarea", "required": False},
            {"id": "home_escapes", "text": "Бывали ли у вас случаи побегов из дома?", "type": "radio", "options": ["Да", "Нет"], "required": True}
        ]
    },
    "social_connections": {
        "title": "🌐 Социальные связи",
        "questions": [
            {"id": "astana_contacts", "text": "Есть ли в городе Астана родственники или знакомые (ФИО и адрес)", "type": "textarea", "required": False},
            {"id": "family_suicides", "text": "Были ли самоубийства или суицидальные попытки у родственников", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_suicides", "text": "Имелись ли у вас в прошлом суицидальные попытки/мысли", "type": "radio", "options": ["Да", "Нет"], "required": True}
        ]
    },
    "health_history": {
        "title": "🏥 Медицинская история",
        "questions": [
            {"id": "family_alcoholism", "text": "Был ли в вашей семье алкоголизм", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "family_drugs", "text": "Была ли в вашей семье наркомания", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "family_criminal", "text": "Была ли в вашей семье судимость", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "family_mental", "text": "Были ли в семье наследственные нервно-психические заболевания", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_alcoholism", "text": "Были ли у вас до армии факты алкоголизма", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_drugs", "text": "Были ли у вас до армии факты наркомании", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_criminal", "text": "Были ли у вас до армии судимости", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_mental", "text": "Были ли у вас до армии нервно-психические заболевания", "type": "radio", "options": ["Да", "Нет"], "required": True},
	        {"id": "personal_headtrauma", "text": "Были ли у Вас сотрясения мозга/травмы головы", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "personal_gambling", "text": "Была ли у вас игромания", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "hereditary_diseases", "text": "Имеешь ли ты тяжёлые наследственные заболевания? (онкологические, дыхательные, гипертония, сердечные и т.д.)", "type": "textarea", "required": False},
            {"id": "seizures", "text": "Были ли у ближайших родственников или у вас судорожные припадки", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "bedwetting", "text": "Было ли у вас ночное недержание мочи?", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "bedwetting_age", "text": "В каком возрасте? (если было недержание)", "type": "number", "required": False}
        ]
    },
    "work_military": {
        "title": "💼 Работа и военная служба",
        "questions": [
            {"id": "work_before_army", "text": "Кем работал до армии, сколько времени?", "type": "textarea", "required": False},
            {"id": "want_serve", "text": "Желаете ли вы проходить военную службу", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "serve_reason", "text": "Причина (если не желаете служить)", "type": "textarea", "required": False},
            {"id": "service_difficulties", "text": "В чем для вас будет трудность воинской службы", "type": "multiselect", "options": ["Беспрекословное подчинение", "Физические нагрузки", "Удаленность от дома", "Высокая личная ответственность", "Преодоление собственных отрицательных привычек", "Другое"], "required": True}
        ]
    },
    "religion_lifestyle": {
        "title": "🕌 Религия и образ жизни",
        "questions": [
            {"id": "religion_type", "text": "Какую религию исповедуешь", "type": "text", "required": False},
            {"id": "religion_direction", "text": "Какое направление религии", "type": "text", "required": False},
            {"id": "religion_teachers", "text": "Если ты слушаешь духовных учителей, то перечисли их", "type": "text", "required": False},
            {"id": "religious_attendance", "text": "Как часто ходишь в мечеть/церковь", "type": "select", "options": ["Каждый день", "Несколько раз в неделю", "Раз в неделю", "Несколько раз в месяц", "Редко", "Никогда"], "required": False},
            {"id": "traditional_holidays", "text": "Празднуете ли вы традиционные праздники?", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "social_events", "text": "Ходите на различные торжества (дни рождения, свадьбы)", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "girlfriend", "text": "Есть ли девушка?", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "relationship_closeness", "text": "Насколько близкие отношения по шкале от 1 до 5", "type": "slider", "min": 1, "max": 5, "required": False},
            {"id": "relationship_period", "text": "Сколько времени вы в отношениях", "type": "text", "required": False}
        ]
    },
    "financial_health": {
        "title": "💰 Финансы и здоровье",
        "questions": [
            {"id": "betting", "text": "Делаешь ли ставки в букмекерских конторах или онлайн", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "credits", "text": "Есть у тебя кредиты/займы (сколько, на какую сумму, кто оплачивает)", "type": "textarea", "required": False},
            {"id": "medical_examination", "text": "При прохождении ВВК в ДДО полностью ли вы прошли обследование у врачей", "type": "radio", "options": ["Да", "Нет"], "required": True},
            {"id": "hidden_health_facts", "text": "Есть ли факты относительно вашего здоровья (диагнозы по которым ранее вас не брали на службу), о которых вы не сказали вашему старшему", "type": "textarea", "required": False}
        ]
    }
}

# Функция для установки API ключа OpenAI
def set_openai_api_key(api_key):
    """Устанавливает API ключ OpenAI как переменную окружения"""
    if AUDIO_AVAILABLE:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False

# Функция для озвучивания текста с использованием OpenAI API
def generate_speech(text, voice="alloy", language="ru"):
    """Генерирует аудио для заданного текста с использованием OpenAI TTS API"""
    if not AUDIO_AVAILABLE:
        return None
        
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_filename = temp_audio.name
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        
        response.stream_to_file(temp_filename)
        
        with open(temp_filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        
        os.unlink(temp_filename)
        return audio_html
    
    except Exception as e:
        st.error(f"Ошибка при генерации аудио: {str(e)}")
        return None

# Функции для работы с состоянием сессии
def initialize_session():
    """Инициализация состояния сессии при первом запуске"""
    defaults = {
        'stage': 'start',
        'questionnaire_stage': 'personal_info',
        'questionnaire_responses': {},
        'responses': {},
        'current_scale': None,
        'current_question_index': 0,
        'questions_order': [],
        'scale_scores': {},
        'risk_levels': {},
        'medium_risk_scales': [],
        'high_risk_scales': [],
        'evaluated_scales': [],
        'detailed_results': {},
        'user_info': {'name': '', 'age': '', 'gender': ''},
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'tts_enabled': False,
        'tts_voice': 'alloy',
        'tts_language': 'ru',
        'last_question_id': None,
        'questionnaire_completed': False,
        'risk_levels_desc': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    """Сброс состояния сессии для нового тестирования"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session()

def save_questionnaire_response(question_id, value):
    """Сохранение ответа на вопрос анкеты"""
    st.session_state.questionnaire_responses[question_id] = value

def save_response(question_id, value):
    """Сохранение ответа на вопрос психологического теста"""
    st.session_state.responses[question_id] = value

def show_start_screen():
    """Отображение начального экрана"""
    st.title("🪖 Система психологического тестирования военнослужащих")
    
    st.markdown("""
    ## Добро пожаловать в систему комплексной оценки готовности к военной службе
    
    Данная система включает:
    
    ### 📋 **Этап 1: Военная анкета**
    - Сбор личной информации
    - Семейная история
    - Военная готовность
    
    ### 🧠 **Этап 2: Психологическое тестирование**
    🎯 **Шкала агрессии** (Басса-Перри)  
    🤝 **Шкала изоляции/депривации** (Д. Рассел)  
    💊 **Шкала соматической депрессии** (Бека)  
    😰 **Шкала тревожности и депрессии** (NUDS)  
    🧘 **Шкала нервно-психической устойчивости**  
    🪖 **Шкала военной адаптации** (специализированная)
    """)
    
    # Кнопка начала
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Начать обследование", use_container_width=True, type="primary"):
            st.session_state.stage = 'questionnaire'
            st.rerun()

def show_questionnaire():
    """Отображение военной анкеты"""
    st.title("🪖 Военная анкета")
    
    st.info("""
    📋 **Заполните анкету перед прохождением психологического тестирования**
    
    Пожалуйста, отвечайте честно и полно. Вся информация конфиденциальна и используется только для оценки готовности к военной службе.
    
    🔴 - обязательные поля
    """)
    
    # Навигация по секциям
    sections = list(MILITARY_QUESTIONNAIRE.keys())
    section_names = [MILITARY_QUESTIONNAIRE[s]["title"] for s in sections]
    
    current_section_index = sections.index(st.session_state.questionnaire_stage)
    
    # Табы для навигации
    selected_tab = st.selectbox(
        "Выберите раздел:",
        section_names,
        index=current_section_index
    )
    
    selected_section = sections[section_names.index(selected_tab)]
    st.session_state.questionnaire_stage = selected_section
    
    # Отображение текущей секции
    section_data = MILITARY_QUESTIONNAIRE[selected_section]
    section_complete = show_questionnaire_section(selected_section, section_data)
    
    # Прогресс заполнения
    completed_sections = 0
    total_sections = len(sections)
    
    for section_key in sections:
        section_questions = MILITARY_QUESTIONNAIRE[section_key]["questions"]
        section_filled = True
        for q in section_questions:
            if q.get("required", False):
                if not st.session_state.questionnaire_responses.get(q["id"]):
                    section_filled = False
                    break
        if section_filled:
            completed_sections += 1
    
    progress = completed_sections / total_sections
    st.progress(progress)
    st.write(f"Прогресс: {completed_sections}/{total_sections} разделов заполнено")
    
    # Навигационные кнопки
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_section_index > 0:
            if st.button("⬅️ Предыдущий раздел"):
                st.session_state.questionnaire_stage = sections[current_section_index - 1]
                st.rerun()
    
    with col2:
        if current_section_index < len(sections) - 1:
            if st.button("➡️ Следующий раздел"):
                st.session_state.questionnaire_stage = sections[current_section_index + 1]
                st.rerun()
    
    with col3:
        if progress == 1.0:
            if st.button("✅ Завершить анкету и перейти к тестированию"):
                st.session_state.questionnaire_completed = True
                st.session_state.stage = 'screening'
                prepare_screening_questions()
                st.rerun()
        else:
            st.button("✅ Завершить анкету", disabled=True)
            st.caption("Заполните все обязательные поля")

def show_questionnaire_section(section_name, section_data):
    """Отображение секции анкеты"""
    st.subheader(section_data["title"])
    
    all_filled = True
    
    for question in section_data["questions"]:
        question_id = question["id"]
        question_text = question["text"]
        question_type = question["type"]
        required = question.get("required", False)
        
        # Получаем сохраненное значение
        saved_value = st.session_state.questionnaire_responses.get(question_id)
        
        # Отображаем вопрос в зависимости от типа
        if question_type == "text":
            value = st.text_input(
                f"{'🔴 ' if required else ''}{question_text}",
                value=saved_value or "",
                key=f"q_{question_id}",
                placeholder="Введите ответ..."
            )
            if value != saved_value:
                save_questionnaire_response(question_id, value)
            if required and not value:
                all_filled = False
                
        elif question_type == "textarea":
            value = st.text_area(
                f"{'🔴 ' if required else ''}{question_text}",
                value=saved_value or "",
                key=f"q_{question_id}",
                height=100,
                placeholder="Введите подробный ответ..."
            )
            if value != saved_value:
                save_questionnaire_response(question_id, value)
            if required and not value:
                all_filled = False
                
        elif question_type == "date":
            try:
                default_date = datetime.strptime(saved_value, "%Y-%m-%d").date() if saved_value else date.today()
            except:
                default_date = date.today()
            
            value = st.date_input(
                f"{'🔴 ' if required else ''}{question_text}",
                value=default_date,
                key=f"q_{question_id}",
                min_value=date(1950, 1, 1),
                max_value=date.today()
            )
            save_questionnaire_response(question_id, str(value))
            
        elif question_type == "select":
            options = question["options"]
            index = 0
            if saved_value and saved_value in options:
                index = options.index(saved_value)
            
            value = st.selectbox(
                f"{'🔴 ' if required else ''}{question_text}",
                options,
                index=index,
                key=f"q_{question_id}"
            )
            save_questionnaire_response(question_id, value)
            if required and not value:
                all_filled = False
                
        elif question_type == "radio":
            options = question["options"]
            index = 0
            if saved_value and saved_value in options:
                index = options.index(saved_value)
            
            value = st.radio(
                f"{'🔴 ' if required else ''}{question_text}",
                options,
                index=index,
                key=f"q_{question_id}",
                horizontal=True
            )
            save_questionnaire_response(question_id, value)
            if required and not value:
                all_filled = False
                
        elif question_type == "multiselect":
            options = question["options"]
            default = saved_value.split(", ") if saved_value else []
            
            value = st.multiselect(
                f"{'🔴 ' if required else ''}{question_text}",
                options,
                default=default,
                key=f"q_{question_id}"
            )
            save_questionnaire_response(question_id, ", ".join(value))
            if required and not value:
                all_filled = False
                
        elif question_type == "number":
            value = st.number_input(
                f"{'🔴 ' if required else ''}{question_text}",
                value=int(saved_value) if saved_value else 0,
                min_value=0,
                max_value=100,
                key=f"q_{question_id}"
            )
            save_questionnaire_response(question_id, str(value))
            
        elif question_type == "slider":
            min_val = question.get("min", 1)
            max_val = question.get("max", 5)
            value = st.slider(
                f"{'🔴 ' if required else ''}{question_text}",
                min_value=min_val,
                max_value=max_val,
                value=int(saved_value) if saved_value else min_val,
                key=f"q_{question_id}"
            )
            save_questionnaire_response(question_id, str(value))
        
        st.markdown("---")
    
    return all_filled

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
    np.random.shuffle(questions)
    st.session_state.questions_order = questions
    st.session_state.current_question_index = 0

def show_question(question, progress=None):
    """Отображение вопроса с шкалой ответов"""
    # Заголовок с прогрессом
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Вопрос {st.session_state.current_question_index + 1} из {len(st.session_state.questions_order)}")
    with col2:
        if progress is not None:
            st.metric("Прогресс", f"{int(progress * 100)}%")
    
    if progress is not None:
        st.progress(progress)
    
    # Отображение вопроса
    st.markdown(f"""
    ### 💭 {question['text']}
    """)
    
    # Озвучивание вопроса если включено
    if st.session_state.tts_enabled and AUDIO_AVAILABLE:
        audio_container = st.empty()
        
        if os.environ.get("OPENAI_API_KEY"):
            # Автоматическое озвучивание при появлении вопроса
            if st.session_state.last_question_id != question['id']:
                audio_html = generate_speech(question['text'], voice=st.session_state.tts_voice, language=st.session_state.tts_language)
                if audio_html:
                    audio_container.markdown(audio_html, unsafe_allow_html=True)
                    st.session_state.last_question_id = question['id']
            
            # Кнопка для повторного озвучивания
            if st.button("🔊 Озвучить еще раз", key="tts_button"):
                audio_html = generate_speech(question['text'], voice=st.session_state.tts_voice, language=st.session_state.tts_language)
                if audio_html:
                    audio_container.markdown(audio_html, unsafe_allow_html=True)
    
    # Создаем кнопки для оценки от 1 до 5
    st.markdown("**Выберите ваш ответ:**")
    
    cols = st.columns(5)
    selected_value = None
    
    button_labels = [
        "1️⃣ Совершенно не согласен",
        "2️⃣ Скорее не согласен", 
        "3️⃣ Нейтрально",
        "4️⃣ Скорее согласен",
        "5️⃣ Полностью согласен"
    ]
    
    for i, col in enumerate(cols):
        value = i + 1
        with col:
            if col.button(button_labels[i], key=f"btn_{question['id']}_{value}", use_container_width=True):
                selected_value = value
    
    # Если выбран ответ, переходим к следующему вопросу
    if selected_value is not None:
        save_response(question['id'], selected_value)

        # Показываем подтверждение выбора
        st.success(f"✅ Ваш ответ записан: {selected_value}")

        # Проверяем, есть ли еще вопросы
        if st.session_state.current_question_index < len(st.session_state.questions_order) - 1:
            st.session_state.current_question_index += 1
        else:
            # Завершение текущего этапа
            if st.session_state.stage == 'screening':
                # Завершаем первичный скрининг
                st.session_state.stage = 'results'
                analyze_results()
            elif st.session_state.stage == 'medium_risk_assessment':
                # Завершаем оценку средней шкалы риска
                complete_detailed_assessment('medium')
            elif st.session_state.stage == 'high_risk_assessment':
                # Завершаем оценку высокой шкалы риска
                complete_detailed_assessment('high')

        st.rerun()
    
    # Показать предыдущий ответ, если есть
    if question['id'] in st.session_state.responses:
        prev_answer = st.session_state.responses[question['id']]
        st.info(f"Ваш предыдущий ответ: {prev_answer} - {button_labels[prev_answer-1].split(' ', 1)[1]}")

def analyze_results():
    """Анализ результатов скрининга и определение уровней риска"""
    scores = {}
    positive_answers = {}  # Счетчик положительных ответов для каждой шкалы
    
    # Расчет баллов и подсчет положительных ответов по каждой шкале
    for scale in SCREENING_QUESTIONS.keys():
        score = 0
        count = 0
        positive_count = 0
        
        for question_id, value in st.session_state.responses.items():
            # Определяем, к какой шкале относится вопрос
            for s, questions in SCREENING_QUESTIONS.items():
                if any(q['id'] == question_id for q in questions):
                    if s == scale:
                        score += value
                        count += 1
                        # Считаем ответ положительным, если значение 4 или 5
                        if value >= 4:
                            positive_count += 1
                    break
        
        if count > 0:
            scores[scale] = score
            positive_answers[scale] = positive_count
    
    st.session_state.scale_scores = scores
    
    # Определение уровней риска
    risk_levels = {}
    risk_levels_desc = {}
    medium_risk_scales = []
    high_risk_scales = []
    
    # Определение уровней риска для каждой шкалы
    for scale, score in scores.items():
        if scale == 'sincerity':
            # Специальная логика для шкалы искренности
            if score >= 13 or score <= 4:
                risk_levels[scale] = "warning"
                risk_levels_desc[scale] = "низкая искренность ответов"
        else:
            # Если 2 или более положительных ответа в скрининге
            if positive_answers.get(scale, 0) >= 2:
                risk_levels[scale] = "medium"
                risk_levels_desc[scale] = "средний уровень риска (требуется дополнительная оценка)"
                medium_risk_scales.append(scale)
            else:
                risk_levels[scale] = "low"
                risk_levels_desc[scale] = "низкий уровень риска"
    
    st.session_state.risk_levels = risk_levels
    st.session_state.risk_levels_desc = risk_levels_desc
    st.session_state.medium_risk_scales = medium_risk_scales
    st.session_state.high_risk_scales = high_risk_scales
    
    # Определение следующего этапа
    if medium_risk_scales:
        # Есть шкалы со средним риском - дополнительные вопросы
        st.session_state.stage = 'medium_risk_assessment'
        st.session_state.current_scale = medium_risk_scales[0]
        prepare_detailed_questions(medium_risk_scales[0], "medium")
    else:
        # Все шкалы в норме - переход к результатам
        st.session_state.stage = 'results'
        prepare_final_recommendations()

def prepare_detailed_questions(scale, risk_level):
    """Подготовка дополнительных вопросов для углубленной оценки"""
    questions = []
    
    if risk_level == "medium" and scale in MEDIUM_RISK_QUESTIONS:
        # Дополнительные вопросы для среднего риска
        for question in MEDIUM_RISK_QUESTIONS[scale]:
            questions.append({
                "id": question["id"],
                "text": question["text"],
                "scale": scale
            })
    elif risk_level == "high" and scale in HIGH_RISK_QUESTIONS:
        # Полный опросник для высокого риска
        for question in HIGH_RISK_QUESTIONS[scale]:
            questions.append({
                "id": question["id"],
                "text": question["text"],
                "scale": scale
            })
    
    # Если нет специальных вопросов, используем вопросы среднего риска
    if not questions and scale in MEDIUM_RISK_QUESTIONS:
        for question in MEDIUM_RISK_QUESTIONS[scale]:
            questions.append({
                "id": question["id"],
                "text": question["text"],
                "scale": scale
            })
    
    np.random.shuffle(questions)
    st.session_state.questions_order = questions
    st.session_state.current_question_index = 0

def prepare_final_recommendations():
    """Подготовка итоговых рекомендаций"""
    recommendations = []
    
    # Проверяем наличие высокого риска в любой шкале
    has_high_risk = False
    for scale, level in st.session_state.risk_levels_desc.items():
        if "высокий" in level:
            has_high_risk = True
            recommendations.append(f"⚠️ Выявлен высокий уровень риска по шкале '{SCALE_NAMES.get(scale, scale)}'")
    
    # Проверка риска экстремизма
    questionnaire = st.session_state.questionnaire_responses
    has_religious_teacher = bool(questionnaire.get('religion_teachers', '').strip())
    frequent_attendance = questionnaire.get('religious_attendance') in ['Несколько раз в неделю', 'Каждый день']
    no_social_events = questionnaire.get('social_events') == 'Нет'

    if has_religious_teacher and frequent_attendance and no_social_events:
        has_high_risk = True
        recommendations.append("⚠️ Выявлен риск экстремизма")
    
    # Если есть хотя бы один высокий риск, добавляем общую рекомендацию
    if has_high_risk:
        recommendations.append("❌ **Не рекомендуется к военной службе**")
        st.session_state.military_recommendation = "not_recommended"
    else:
        # Проверяем наличие среднего риска
        medium_risk_scales = []
        for scale, level in st.session_state.risk_levels_desc.items():
            if "средний" in level:
                medium_risk_scales.append(SCALE_NAMES.get(scale, scale))
        
        if medium_risk_scales:
            recommendations.append(f"⚠️ Требуется дополнительное внимание к следующим аспектам: {', '.join(medium_risk_scales)}")
            recommendations.append("✅ **Рекомендуется к военной службе с ограничениями**")
            st.session_state.military_recommendation = "recommended_with_restrictions"
        else:
            recommendations.append("✅ **Рекомендуется к военной службе**")
            st.session_state.military_recommendation = "recommended"
    
    st.session_state.final_recommendations = recommendations

def complete_detailed_assessment(risk_level):
    """Завершение углубленной оценки для текущей шкалы"""
    current_scale = st.session_state.current_scale
    
    # Подсчет результатов углубленной оценки
    detailed_score = 0
    question_count = 0
    positive_count = 0
    
    for question_id, value in st.session_state.responses.items():
        # Проверяем, относится ли вопрос к текущей углубленной оценке
        if risk_level == "medium" and current_scale in MEDIUM_RISK_QUESTIONS:
            if any(q['id'] == question_id for q in MEDIUM_RISK_QUESTIONS[current_scale]):
                detailed_score += value
                question_count += 1
                if value >= 4:  # Считаем ответ положительным, если значение 4 или 5
                    positive_count += 1
        elif risk_level == "high" and current_scale in HIGH_RISK_QUESTIONS:
            if any(q['id'] == question_id for q in HIGH_RISK_QUESTIONS[current_scale]):
                detailed_score += value
                question_count += 1
                if value >= 4:  # Считаем ответ положительным, если значение 4 или 5
                    positive_count += 1
    
    # Сохранение результатов углубленной оценки
    if question_count > 0:
        max_possible = question_count * 5  # Максимальный балл (5 баллов за вопрос)
        percentage = (detailed_score / max_possible) * 100
        
        st.session_state.detailed_results[current_scale] = {
            'score': detailed_score,
            'max_possible': max_possible,
            'percentage': percentage,
            'positive_count': positive_count,
            'total_questions': question_count
        }
        
        # Обновление уровня риска на основе углубленной оценки
        if risk_level == "medium":
            # Если более 50% ответов положительные, переходим к высокой оценке риска
            if positive_count > question_count / 2:
                st.session_state.risk_levels_desc[current_scale] = "высокий уровень риска (подтверждено углубленной оценкой)"
                st.session_state.high_risk_scales.append(current_scale)
                # Переходим к высокой оценке риска
                st.session_state.stage = 'high_risk_assessment'
                prepare_detailed_questions(current_scale, "high")
                return
            else:
                st.session_state.risk_levels_desc[current_scale] = "средний уровень риска (подтверждено углубленной оценкой)"
        elif risk_level == "high":
            if percentage >= 70:
                st.session_state.risk_levels_desc[current_scale] = "высокий уровень риска (подтверждено углубленной оценкой)"
            else:
                st.session_state.risk_levels_desc[current_scale] = "средний уровень риска (скорректировано после углубленной оценки)"
    
    # Отмечаем шкалу как оцененную
    if hasattr(st.session_state, 'evaluated_scales'):
        st.session_state.evaluated_scales.append(current_scale)
    else:
        st.session_state.evaluated_scales = [current_scale]
    
    # Определяем следующий этап
    if risk_level == "high":
        # Убираем текущую шкалу из списка high_risk_scales
        remaining_high_risk = [s for s in st.session_state.high_risk_scales if s != current_scale]
        st.session_state.high_risk_scales = remaining_high_risk
        
        if remaining_high_risk:
            # Есть еще шкалы с высоким риском
            st.session_state.current_scale = remaining_high_risk[0]
            prepare_detailed_questions(remaining_high_risk[0], "high")
        else:
            # Переходим к результатам
            st.session_state.stage = 'results'
            prepare_final_recommendations()
    
    elif risk_level == "medium":
        # Убираем текущую шкалу из списка medium_risk_scales
        remaining_medium_risk = [s for s in st.session_state.medium_risk_scales if s != current_scale]
        remaining_medium_risk = [s for s in remaining_medium_risk if s not in st.session_state.evaluated_scales]
        st.session_state.medium_risk_scales = remaining_medium_risk
        
        if remaining_medium_risk:
            # Есть еще шкалы со средним риском
            st.session_state.current_scale = remaining_medium_risk[0]
            prepare_detailed_questions(remaining_medium_risk[0], "medium")
        else:
            # Проверяем, есть ли шкалы с высоким риском
            if st.session_state.high_risk_scales:
                st.session_state.stage = 'high_risk_assessment'
                st.session_state.current_scale = st.session_state.high_risk_scales[0]
                prepare_detailed_questions(st.session_state.high_risk_scales[0], "high")
            else:
                # Переходим к результатам
                st.session_state.stage = 'results'
                prepare_final_recommendations()

def show_screening():
    """Отображение экрана первичного скрининга"""
    st.title("🔍 Психологический скрининг")
    
    st.markdown("""
    Пожалуйста, ответьте на следующие вопросы, выбирая оценку от 1 до 5.
    
    📝 **Помните**: Нет правильных или неправильных ответов. Отвечайте искренне, основываясь на том, как вы себя чувствуете в последнее время.
    """)
    
    # Вычисляем прогресс
    progress = st.session_state.current_question_index / len(st.session_state.questions_order)
    
    # Отображаем текущий вопрос
    question = st.session_state.questions_order[st.session_state.current_question_index]
    show_question(question, progress)

def show_sincerity_warning():
    """Отображение предупреждения о возможной недостоверности ответов"""
    st.title("⚠️ Внимание: Возможная недостоверность ответов")
    
    st.warning("""
    ## 🚨 Обратите внимание
    
    Система обнаружила возможную недостоверность в ваших ответах. Это может повлиять на точность оценки готовности к военной службе.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Однообразность ответов**
        
        Вы могли отвечать слишком однообразно
        """)
    
    with col2:
        st.markdown("""
        **❓ Непонимание вопросов**
        
        Возможно, некоторые формулировки были неясны
        """)
    
    with col3:
        st.markdown("""
        **🎭 Социальная желательность**
        
        Стремление представить себя в лучшем свете
        """)
    
    st.info("💡 **Для получения корректной оценки важна максимальная честность.**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Пройти тестирование заново", use_container_width=True):
            # Сохраняем анкету, но сбрасываем тест
            questionnaire_data = st.session_state.questionnaire_responses.copy()
            reset_session()
            st.session_state.questionnaire_responses = questionnaire_data
            st.session_state.questionnaire_completed = True
            st.session_state.stage = 'screening'
            prepare_screening_questions()
            st.rerun()
    
    with col2:
        if st.button("➡️ Продолжить с текущими ответами", use_container_width=True):
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
                prepare_final_recommendations()
            
            st.rerun()

def show_detailed_assessment():
    """Отображение экрана углубленной оценки"""
    current_scale = st.session_state.current_scale
    scale_name = SCALE_NAMES.get(current_scale, current_scale)
    risk_level = "средним" if st.session_state.stage == 'medium_risk_assessment' else "высоким"
    
    st.title(f"🎯 Углубленная оценка")
    
    # Информационная панель
    st.info(f"""
    **📊 {scale_name}**
    
    По результатам первичного скрининга выявлен **{risk_level} риск** по данной шкале.
    
    Пожалуйста, ответьте на дополнительные вопросы для более точной оценки.
    """)
    
    # Вычисляем прогресс
    progress = st.session_state.current_question_index / len(st.session_state.questions_order)
    
    # Отображаем текущий вопрос
    question = st.session_state.questions_order[st.session_state.current_question_index]
    show_question(question, progress)

def show_results():
    """Отображение результатов тестирования"""
    st.title("📊 Результаты обследования военнослужащего")
    
    # Заголовок с информацией о кандидате
    questionnaire = st.session_state.questionnaire_responses

    # Автосохранение результата в БД (один раз за сессию)
    if not st.session_state.get('saved_to_db'):
        try:
            from db import save_assessment
            payload = {
                'session_id': st.session_state.session_id,
                'questionnaire_responses': st.session_state.questionnaire_responses,
                'responses': st.session_state.responses,
                'scale_scores': st.session_state.scale_scores,
                'risk_levels': st.session_state.risk_levels,
                'risk_levels_desc': st.session_state.risk_levels_desc,
                'detailed_results': st.session_state.detailed_results,
                'military_recommendation': st.session_state.get('military_recommendation'),
                'final_recommendations': st.session_state.get('final_recommendations'),
            }
            st.session_state.saved_assessment_id = save_assessment(payload)
            st.session_state.saved_to_db = True
        except Exception as e:
            st.warning(f"⚠️ Результат не сохранён в базу данных ({e}). Скачайте отчёт вручную ниже.")

    st.markdown(f"""
    ## 👤 Информация о кандидате
    
    **ФИО**: {questionnaire.get('full_name', 'Не указано')}  
    **Дата рождения**: {questionnaire.get('birth_date', 'Не указано')}  
    **Национальность**: {questionnaire.get('nationality', 'Не указано')}  
    **Образование**: {questionnaire.get('education', 'Не указано')}  
    **Семейное положение**: {questionnaire.get('marital_status', 'Не указано')}
    """)
    
    st.markdown("---")
    
    # Предупреждение о достоверности, если было
    if st.session_state.risk_levels.get('sincerity') == "warning_ignored":
        st.error("⚠️ **ВНИМАНИЕ**: Результаты психологического тестирования могут иметь сниженную достоверность из-за выявленных особенностей в ответах.")
    
    # Общая оценка готовности к службе
    st.subheader("🪖 Общая оценка готовности к военной службе")
    
    # Анализ ключевых показателей
    critical_issues = []
    warning_issues = []
    
    # Проверка критических факторов
    if questionnaire.get("family_suicides") == "Да" or questionnaire.get("personal_suicides") == "Да":
        critical_issues.append("Суицидальные факторы")
    
    if questionnaire.get("want_serve") == "Нет":
        critical_issues.append("Нежелание служить")
    
    high_risk_scales = [scale for scale, level in st.session_state.risk_levels_desc.items() if "высокий" in level]
    if high_risk_scales:
        critical_issues.append(f"Высокие психологические риски ({len(high_risk_scales)} шкал)")
    
    # Проверка предупреждающих факторов
    dependencies = []
    if questionnaire.get("family_alcoholism") == "Да" or questionnaire.get("personal_alcoholism") == "Да":
        dependencies.append("алкоголизм")
    if questionnaire.get("family_drugs") == "Да" or questionnaire.get("personal_drugs") == "Да":
        dependencies.append("наркомания")
    if questionnaire.get("personal_gambling") == "Да" or questionnaire.get("betting") == "Да":
        dependencies.append("игровая зависимость")
    
    if dependencies:
        warning_issues.append(f"Факторы зависимости: {', '.join(dependencies)}")
    
    if questionnaire.get("credits"):
        warning_issues.append("Финансовые обязательства")
    
    if questionnaire.get("hidden_health_facts"):
        warning_issues.append("Скрытые медицинские факты")
    
    # Add this to the critical factors check section
    if (bool(questionnaire.get('religion_teachers', '').strip()) and 
        questionnaire.get('religious_attendance') in ['Несколько раз в неделю', 'Каждый день'] and 
        questionnaire.get('social_events') == 'Нет'):
        critical_issues.append("Риск экстремизма")
    
    # Итоговое заключение
    if critical_issues:
        st.error(f"""
        **❌ Не рекомендуется к военной службе**
        
        Выявлены критические факторы риска:
        {chr(10).join('• ' + issue for issue in critical_issues)}
        """)
        recommendation_color = "🔴"
        final_recommendation = "Не рекомендуется к военной службе"
    elif warning_issues:
        st.warning(f"""
        **⚠️ УСЛОВНО ГОДЕН к военной службе**
        
        Выявлены факторы, требующие внимания:
        {chr(10).join('• ' + issue for issue in warning_issues)}
        
        Требуется дополнительная работа и наблюдение.
        """)
        recommendation_color = "🟡"
        final_recommendation = "УСЛОВНО ГОДЕН"
    else:
        st.success("""
        **✅ РЕКОМЕНДУЕТСЯ к военной службе**
        
        Серьезных противопоказаний не выявлено.
        Кандидат готов к прохождению военной службы.
        """)
        recommendation_color = "🟢"
        final_recommendation = "РЕКОМЕНДУЕТСЯ"
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Заключение", final_recommendation)
    with col2:
        st.metric("Критические риски", len(critical_issues))
    with col3:
        st.metric("Предупреждения", len(warning_issues))
    with col4:
        st.metric("Вопросов отвечено", len(st.session_state.responses))
    
    st.markdown("---")
    
    # Детальные результаты психологического тестирования
    st.subheader("🧠 Результаты психологического тестирования")
    
    # Отображение графика результатов только если есть данные
    if st.session_state.scale_scores:
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
            max_scores.append(15)  # Максимум для первичного скрининга - 15 баллов
            
            # Определение цвета в зависимости от уровня риска
            if score <= THRESHOLDS["low"][1]:
                colors.append('#4CAF50')  # Зеленый для низкого риска
            elif score <= THRESHOLDS["medium"][1]:
                colors.append('#FFC107')  # Желтый для среднего риска
            else:
                colors.append('#F44336')  # Красный для высокого риска
        
        # Данные из углубленной оценки
        for scale, result in st.session_state.detailed_results.items():
            scale_name = f"{SCALE_NAMES.get(scale, scale)} (углубленная оценка)"
            
            scales.append(scale_name)
            scores.append(result['score'])
            max_scores.append(result['max_possible'])
            
            # Определение цвета в зависимости от процента
            percentage = result['percentage']
            if percentage < 33:
                colors.append('#4CAF50')  # Зеленый
            elif percentage < 67:
                colors.append('#FFC107')  # Желтый
            else:
                colors.append('#F44336')  # Красный
        
        if scales:  # Проверяем, что есть данные для отображения
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
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Создаем горизонтальную гистограмму
            bars = ax.barh(df['Шкала'], df['Процент'], color=df['Цвет'], alpha=0.8)
            
            # Добавление подписей значений
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 1
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{df["Балл"].iloc[i]}/{df["Максимум"].iloc[i]} ({width:.1f}%)',
                       va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Процент от максимального значения', fontsize=12)
            ax.set_title('Результаты психологического тестирования', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlim(0, 110)
            
            # Добавление линий уровней риска
            ax.axvline(x=33.33, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=66.67, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Добавление области риска
            ax.axvspan(0, 33.33, alpha=0.1, color='green', label='Низкий риск')
            ax.axvspan(33.33, 66.67, alpha=0.1, color='yellow', label='Средний риск')
            ax.axvspan(66.67, 100, alpha=0.1, color='red', label='Высокий риск')
            
            ax.grid(axis='x', alpha=0.3)
            ax.legend(loc='lower right')
            
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            
            st.pyplot(fig)
    
    # Подробное описание результатов
    st.subheader("📋 Детальный анализ по шкалам")
    
    if st.session_state.risk_levels_desc:
        for scale, level in st.session_state.risk_levels_desc.items():
            scale_name = SCALE_NAMES.get(scale, scale)
            
            if "высокий" in level:
                emoji = "🔴"
            elif "средний" in level:
                emoji = "🟡"
            else:
                emoji = "🟢"
            
            with st.container():
                st.markdown(f"### {emoji} {scale_name}")
                st.markdown(f"**Уровень риска**: {level}")
                
                # Специфичные описания для военной службы
                if scale == "aggression":
                    if "высокий" in level:
                        st.error("⚠️ **ВОЕННЫЙ РИСК**: Высокая агрессивность может привести к конфликтам в воинском коллективе и нарушениям дисциплины.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Умеренная агрессивность. Требуется контроль в стрессовых ситуациях.")
                    else:
                        st.success("✅ **НОРМА**: Агрессивность в пределах нормы. Хороший самоконтроль.")
                
                elif scale == "military_adaptation":
                    if "высокий" in level:
                        st.error("⚠️ **КРИТИЧНО**: Серьезные проблемы с адаптацией к военной службе. Требуется дополнительная подготовка.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Могут возникнуть трудности адаптации. Рекомендуется усиленная поддержка.")
                    else:
                        st.success("✅ **ОТЛИЧНО**: Хорошие предпосылки для успешной военной службы.")
                
                elif scale == "isolation":
                    if "высокий" in level:
                        st.error("⚠️ **ВОЕННЫЙ РИСК**: Склонность к изоляции может усугубиться в условиях казармы.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Периодическое чувство одиночества. Важна социальная поддержка.")
                    else:
                        st.success("✅ **НОРМА**: Хорошие социальные навыки для командной работы.")
                
                elif scale == "anxiety":
                    if "высокий" in level:
                        st.error("⚠️ **ВОЕННЫЙ РИСК**: Высокая тревожность может привести к срывам в экстремальных ситуациях.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Умеренная тревожность. Рекомендуются техники стресс-менеджмента.")
                    else:
                        st.success("✅ **НОРМА**: Хорошая стрессоустойчивость.")
                
                elif scale == "stability":
                    if "высокий" in level:
                        st.error("⚠️ **КРИТИЧНО**: Низкая психическая устойчивость несовместима с военной службой.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Возможны трудности в стрессовых ситуациях.")
                    else:
                        st.success("✅ **ОТЛИЧНО**: Высокая стрессоустойчивость.")
                
                elif scale == "somatic":
                    if "высокий" in level:
                        st.error("⚠️ **МЕДИЦИНСКИЙ РИСК**: Соматические проявления стресса могут усугубиться в армии.")
                    elif "средний" in level:
                        st.warning("⚠️ **ВНИМАНИЕ**: Периодические соматические симптомы стресса.")
                    else:
                        st.success("✅ **НОРМА**: Хорошая устойчивость к соматическим проявлениям стресса.")
                
                st.markdown("---")
    
    # Заключение специалиста
    st.subheader("📄 Заключение психолога")
    
    conclusion_template = f"""
**ПСИХОЛОГИЧЕСКОЕ ЗАКЛЮЧЕНИЕ**

Кандидат: {questionnaire.get('full_name', 'Не указано')}
Дата обследования: {datetime.now().strftime('%d.%m.%Y')}

**ИТОГОВОЕ ЗАКЛЮЧЕНИЕ**: {recommendation_color} {final_recommendation}

**Обоснование**:
"""
    
    if critical_issues:
        conclusion_template += f"\nВыявлены критические факторы риска: {', '.join(critical_issues)}"
    
    if warning_issues:
        conclusion_template += f"\nВыявлены предупреждающие факторы: {', '.join(warning_issues)}"
    
    if not critical_issues and not warning_issues:
        conclusion_template += "\nСерьезных противопоказаний к военной службе не выявлено."
    
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        conclusion_template += f"""

**Рекомендации**:
{chr(10).join('• ' + rec.split('] ', 1)[-1] if '] ' in rec else rec for rec in st.session_state.recommendations[:3])}

Психолог: ________________
Дата: {datetime.now().strftime('%d.%m.%Y')}
"""
    
    st.text_area("", value=conclusion_template, height=300, disabled=True)
    
    # Действия с результатами
    st.subheader("💾 Экспорт результатов")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Полный отчет (TXT)", use_container_width=True):
            try:
                report_text = generate_military_report()
                st.download_button(
                    label="📥 Скачать полный отчет",
                    data=report_text,
                    file_name=f"military_assessment_{questionnaire.get('full_name', 'candidate')}_{st.session_state.session_id}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Ошибка при генерации отчета: {str(e)}")
    
    with col2:
        if st.button("📊 Данные (CSV)", use_container_width=True):
            try:
                csv_data = generate_military_csv()
                st.download_button(
                    label="📥 Скачать данные",
                    data=csv_data,
                    file_name=f"military_data_{questionnaire.get('full_name', 'candidate')}_{st.session_state.session_id}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ошибка при генерации CSV: {str(e)}")
    
    with col3:
        if st.button("🔄 Новое обследование", use_container_width=True):
            reset_session()
            st.rerun()
    
    # Предупреждение
    st.warning("""
    ### ⚕️ Важная информация
    
    **Данное обследование носит предварительный характер.**
    
    - Результаты должны рассматриваться в комплексе с медицинским обследованием
    - При выявлении серьезных рисков требуется консультация специалиста
    - Окончательное решение принимается военно-врачебной комиссией
    
    📞 **При кризисных ситуациях**: 8-800-2000-122 (психологическая помощь)
    """)

def generate_military_report():
    """Генерация полного военного отчета"""
    questionnaire = st.session_state.questionnaire_responses
    report = []
    
    report.append("ОТЧЕТ О ПСИХОЛОГИЧЕСКОМ ОБСЛЕДОВАНИИ ВОЕННОСЛУЖАЩЕГО")
    report.append("=" * 60)
    report.append(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    report.append(f"ID обследования: {st.session_state.session_id}")
    report.append("")
    
    # Личная информация
    report.append("ЛИЧНАЯ ИНФОРМАЦИЯ:")
    report.append("-" * 20)
    report.append(f"ФИО: {questionnaire.get('full_name', 'Не указано')}")
    report.append(f"Дата рождения: {questionnaire.get('birth_date', 'Не указано')}")
    report.append(f"Место рождения: {questionnaire.get('birth_place', 'Не указано')}")
    report.append(f"Место жительства: {questionnaire.get('residence', 'Не указано')}")
    report.append(f"Национальность: {questionnaire.get('nationality', 'Не указано')}")
    report.append(f"Образование: {questionnaire.get('education', 'Не указано')}")
    report.append(f"Семейное положение: {questionnaire.get('marital_status', 'Не указано')}")
    report.append("")
    
    # Результаты психологического тестирования
    if hasattr(st.session_state, 'scale_scores') and st.session_state.scale_scores:
        report.append("РЕЗУЛЬТАТЫ ПСИХОЛОГИЧЕСКОГО ТЕСТИРОВАНИЯ:")
        report.append("-" * 45)
        
        for scale, score in st.session_state.scale_scores.items():
            if scale == 'sincerity':
                continue
            scale_name = SCALE_NAMES.get(scale, scale)
            level = st.session_state.risk_levels_desc.get(scale, 'не определено')
            report.append(f"{scale_name}: {score} баллов - {level}")
        
        report.append("")
    
    # Рекомендации
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        report.append("РЕКОМЕНДАЦИИ:")
        report.append("-" * 15)
        for i, rec in enumerate(st.session_state.recommendations, 1):
            clean_rec = ''.join(char for char in rec if ord(char) < 128 or char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ')
            report.append(f"{i}. {clean_rec}")
        
        report.append("")
    
    # Заключение
    critical_issues = []
    if questionnaire.get("family_suicides") == "Да" or questionnaire.get("personal_suicides") == "Да":
        critical_issues.append("Суицидальные факторы")
    if questionnaire.get("want_serve") == "Нет":
        critical_issues.append("Нежелание служить")
    
    high_risk_scales = []
    if hasattr(st.session_state, 'risk_levels_desc'):
        high_risk_scales = [scale for scale, level in st.session_state.risk_levels_desc.items() if "высокий" in level]
    if high_risk_scales:
        critical_issues.append(f"Высокие психологические риски")
    
    report.append("ЗАКЛЮЧЕНИЕ:")
    report.append("-" * 12)
    if critical_issues:
        report.append("НЕ РЕКОМЕНДУЕТСЯ к военной службе")
        report.append("Критические факторы риска:")
        for issue in critical_issues:
            report.append(f"- {issue}")
    else:
        report.append("РЕКОМЕНДУЕТСЯ к военной службе при соблюдении мер поддержки")
    
    report.append("")
    report.append("Психолог: ________________")
    report.append(f"Дата: {datetime.now().strftime('%d.%m.%Y')}")
    
    return "\n".join(report)

def generate_military_csv():
    """Генерация CSV с военными данными"""
    questionnaire = st.session_state.questionnaire_responses
    data = []
    
    # Заголовки
    data.append(['Параметр', 'Значение', 'Категория'])
    
    # Личная информация
    personal_data = [
        ('ФИО', questionnaire.get('full_name', ''), 'Личные данные'),
        ('Дата рождения', questionnaire.get('birth_date', ''), 'Личные данные'),
        ('Место рождения', questionnaire.get('birth_place', ''), 'Личные данные'),
        ('Национальность', questionnaire.get('nationality', ''), 'Личные данные'),
        ('Образование', questionnaire.get('education', ''), 'Личные данные'),
        ('Семейное положение', questionnaire.get('marital_status', ''), 'Личные данные')
    ]
    
    for item in personal_data:
        data.append(list(item))
    
    # Медицинские факторы
    medical_data = [
        ('Алкоголизм в семье', questionnaire.get('family_alcoholism', ''), 'Медицинские факторы'),
        ('Наркомания в семье', questionnaire.get('family_drugs', ''), 'Медицинские факторы'),
        ('Психические заболевания в семье', questionnaire.get('family_mental', ''), 'Медицинские факторы'),
        ('Суициды в семье', questionnaire.get('family_suicides', ''), 'Медицинские факторы'),
        ('Личные суицидальные мысли', questionnaire.get('personal_suicides', ''), 'Медицинские факторы'),
        ('Желание служить', questionnaire.get('want_serve', ''), 'Военная готовность')
    ]
    
    for item in medical_data:
        data.append(list(item))
    
    # Психологические результаты
    if hasattr(st.session_state, 'scale_scores') and hasattr(st.session_state, 'risk_levels_desc'):
        for scale in st.session_state.scale_scores.keys():
            if scale == 'sincerity':
                continue
            scale_name = SCALE_NAMES.get(scale, scale)
            score = st.session_state.scale_scores.get(scale, 0)
            level = st.session_state.risk_levels_desc.get(scale, 'не определено')
            data.append([f"{scale_name} (баллы)", score, 'Психологические результаты'])
            data.append([f"{scale_name} (уровень)", level, 'Психологические результаты'])
    
    # Конвертируем в CSV строку
    import io
    output = io.StringIO()
    import csv
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()

# Основная функция приложения
def show_archive():
    """Просмотр сохранённых обследований (для администратора)."""
    st.title("🗄️ Архив обследований")
    if st.button("⬅️ Назад"):
        st.session_state.stage = 'start'
        st.rerun()
    try:
        from db import list_assessments
        rows = list_assessments()
    except Exception as e:
        st.error(f"Не удалось загрузить архив: {e}")
        return
    if not rows:
        st.info("Архив пуст — пока нет сохранённых обследований.")
        return
    table = [{
        "ID": r.id,
        "Дата": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
        "ФИО": r.full_name or "—",
        "Заключение": r.military_recommendation or "—",
    } for r in rows]
    st.metric("Всего обследований", len(rows))
    st.dataframe(table, use_container_width=True)


def main():
    # Инициализация состояния сессии
    initialize_session()
    
    # Боковая панель с информацией о прогрессе
    with st.sidebar:
        st.header("📋 Прогресс обследования")
        
        # Индикаторы этапов
        stages = [
            ("📝 Анкета", st.session_state.questionnaire_completed),
            ("🧠 Скрининг", st.session_state.stage in ['medium_risk_assessment', 'high_risk_assessment', 'results']),
            ("🎯 Углубленная оценка", st.session_state.stage == 'results' and st.session_state.detailed_results),
            ("📊 Результаты", st.session_state.stage == 'results')
        ]
        
        for stage_name, completed in stages:
            if completed:
                st.success(f"✅ {stage_name}")
            else:
                st.info(f"⏳ {stage_name}")
        
        st.markdown("---")
        
        # Информация о текущем этапе
        if st.session_state.stage == 'questionnaire':
            st.info("Заполняется военная анкета")
            if st.session_state.questionnaire_responses:
                filled_questions = len([q for q in st.session_state.questionnaire_responses.values() if q])
                st.metric("Ответов дано", filled_questions)
        
        elif st.session_state.stage in ['screening', 'medium_risk_assessment', 'high_risk_assessment']:
            st.info("Проходит психологическое тестирование")
            if st.session_state.questions_order:
                progress = (st.session_state.current_question_index + 1) / len(st.session_state.questions_order)
                st.progress(progress)
                st.metric("Прогресс", f"{st.session_state.current_question_index + 1}/{len(st.session_state.questions_order)}")
        
        elif st.session_state.stage == 'results':
            st.success("Обследование завершено")
            if st.session_state.questionnaire_responses.get('full_name'):
                st.write(f"**Кандидат**: {st.session_state.questionnaire_responses['full_name']}")
        
        st.markdown("---")
        
        # Кнопки управления
        if st.button("🔄 Начать заново"):
            reset_session()
            st.rerun()
        
        if st.session_state.stage != 'start' and st.button("🏠 В начало"):
            st.session_state.stage = 'start'
            st.rerun()

        # Архив обследований — виден только если задан ADMIN_PASSWORD
        if os.environ.get("ADMIN_PASSWORD"):
            with st.expander("🗄️ Архив обследований"):
                admin_pwd = st.text_input("Пароль администратора", type="password", key="admin_pwd")
                if admin_pwd:
                    if admin_pwd == os.environ.get("ADMIN_PASSWORD"):
                        if st.button("Открыть архив"):
                            st.session_state.stage = 'archive'
                            st.rerun()
                    else:
                        st.error("Неверный пароль")

        # Информация о системе
        with st.expander("ℹ️ О системе"):
            st.markdown("""
            **Система военного психологического тестирования**
            
            - Адаптивный алгоритм оценки
            - Комплексный анализ анкеты и тестов
            - Специализация на военной службе
            - Автоматическое формирование заключений
            
            **Версия**: 1.0  
            **Разработано**: 2025
            """)
    
    # Отображение соответствующего экрана в зависимости от текущего этапа
    if st.session_state.stage == 'start':
        show_start_screen()
    elif st.session_state.stage == 'questionnaire':
        show_questionnaire()
    elif st.session_state.stage == 'screening':
        show_screening()
    elif st.session_state.stage == 'sincerity_warning':
        show_sincerity_warning()
    elif st.session_state.stage == 'medium_risk_assessment' or st.session_state.stage == 'high_risk_assessment':
        show_detailed_assessment()
    elif st.session_state.stage == 'results':
        show_results()
    elif st.session_state.stage == 'archive':
        show_archive()
    else:
        st.error("❌ Неизвестный этап обследования. Пожалуйста, начните заново.")
        if st.button("🔄 Начать заново"):
            reset_session()
            st.rerun()

# Запуск приложения
if __name__ == "__main__":
    main()