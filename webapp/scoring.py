# -*- coding: utf-8 -*-
"""
Framework-agnostic психологический скоринг военнослужащих.

Чистый модуль без Streamlit / pandas / matplotlib / session_state.
Содержит банки вопросов, названия шкал, структуру военной анкеты
и чистые функции анализа, перенесённые из app.py.

Все функции принимают обычные dict с ответами и возвращают dict.
"""

# ---------------------------------------------------------------------------
# КОНСТАНТЫ / ПОРОГИ
# ---------------------------------------------------------------------------

# Пороги для интерпретации сырого балла шкалы (3 вопроса * 1..5 = 3..15).
# Перенесено из app.py THRESHOLDS. Используется при визуализации цвета.
THRESHOLDS = {
    "low": (3, 7),
    "medium": (8, 11),
    "high": (12, 15),
}

# ---------------------------------------------------------------------------
# БАНКИ ВОПРОСОВ
# ---------------------------------------------------------------------------

# Скрининговые вопросы (первичная оценка): scale_key -> list of {"id", "text"}
SCREENING_QUESTIONS = {
    "aggression": [
        {"id": "ag1", "text": "Я раздражаюсь, когда у меня что-то не получается."},
        {"id": "ag2", "text": "Иногда, когда я неважно себя чувствую, я бываю раздражительным."},
        {"id": "ag3", "text": "Некоторые мои друзья считают, что я вспыльчив."},
    ],
    "isolation": [
        {"id": "is1", "text": "Мне трудно заводить друзей."},
        {"id": "is2", "text": "Мне не хватает общения."},
        {"id": "is3", "text": "Мне не с кем поговорить."},
    ],
    "somatic": [
        {"id": "som1", "text": "Иногда у меня бывает ускоренное сердцебиение"},
        {"id": "som2", "text": "Иногда я чувствую, что я не могу контролировать свои мысли"},
        {"id": "som3", "text": "Иногда у меня бывают желудочно-кишечные расстройства "},
    ],
    "anxiety": [
        {"id": "anx1", "text": "Я испытываю напряженность, мне не по себе"},
        {"id": "anx2", "text": "Приступы плохого настроения у меня бывают редко."},
        {"id": "anx3", "text": "Иногда совершенно безо всякой причины у меня вдруг наступает период необычайной веселости."},
    ],
    "stability": [
        {"id": "stab1", "text": "Я могу получить удовольствие от хорошей книги, радио- или телепрограммы "},
        {"id": "stab2", "text": "Бывало, что при обсуждении некоторых вопросов я, особенно не задумываясь, соглашался с мнением других."},
        {"id": "stab3", "text": "У меня часто бывают подъемы и спады настроения."},
    ],
    "military_adaptation": [
        {"id": "mil1", "text": "Мне трудно выполнять приказы без объяснения причин."},
        {"id": "mil2", "text": "Я боюсь физических нагрузок и испытаний."},
        {"id": "mil3", "text": "Мне сложно находиться далеко от дома длительное время."},
    ],
    "sincerity": [
        {"id": "sin1", "text": "Бывало, что я говорил о вещах, в которых не разбираюсь."},
        {"id": "sin2", "text": "Бывает, что я сержусь."},
        {"id": "sin3", "text": "Иногда я говорю неправду."},
    ],
}

# Дополнительные вопросы для шкал со средним риском
MEDIUM_RISK_QUESTIONS = {
    "aggression": [
        {"id": "ag_med1", "text": "Я дерусь чаще, чем окружающие."},
        {"id": "ag_med2", "text": "Если кто-то ударит меня, я дам сдачи."},
        {"id": "ag_med3", "text": "Иногда я выхожу из себя без особой причины."},
        {"id": "ag_med4", "text": "Мне трудно сдерживать раздражение."},
        {"id": "ag_med5", "text": "Иногда я настолько выходил из себя, что ломал вещи."},
    ],
    "isolation": [
        {"id": "is_med1", "text": "Счастливей всего я бываю, когда я один."},
        {"id": "is_med2", "text": "Если бы люди не были настроены против меня, я достиг бы в жизни гораздо большего."},
        {"id": "is_med3", "text": "Иногда я бываю, уверен, что другие люди знают, о чем я думаю."},
        {"id": "is_med4", "text": "Мне кажется, что по отношению именно ко мне особенно часто поступают несправедливо."},
        {"id": "is_med5", "text": "Часто, даже когда все складывается для меня хорошо, я чувствую, что мне все безразлично."},
        {"id": "is_med6", "text": "Мне кажется, что я все чувствую более остро, чем другие."},
    ],
    "somatic": [
        {"id": "som_med1", "text": "Бывало, что я целыми днями или даже неделями ничего не мог делать, потому что никак не мог заставить себя взяться за работу."},
        {"id": "som_med2", "text": "Иногда я чувствую, что у меня удушье"},
        {"id": "som_med3", "text": "Иногда я чувствую, что у меня затрудненное дыхание"},
        {"id": "som_med4", "text": "Когда я пытаюсь что-то сделать, то часто замечаю, что у меня дрожат руки."},
        {"id": "som_med5", "text": "Иногда я чувствую Испуг"},
        {"id": "som_med6", "text": "Беспокойные мысли крутятся у меня в голове"},
    ],
    "anxiety": [
        {"id": "anx_med1", "text": "у меня бывает внезапное чуство паники"},
        {"id": "anx_med2", "text": "Я испытываю внутреннее напряжение или дрожь"},
        {"id": "anx_med3", "text": "Я испытываю неусидчивость, словно мне постоянно нужно двигаться"},
        {"id": "anx_med4", "text": "То, что приносило мне большое удовольствие, и сейчас вызывает у меня такое же чувство"},
        {"id": "anx_med5", "text": "Работа, требующая пристального внимания, мне нравится."},
    ],
    "stability": [
        {"id": "stab_med1", "text": "Определенно судьба не благосклонна ко мне."},
        {"id": "stab_med2", "text": "Я легко теряю терпение с людьми."},
        {"id": "stab_med3", "text": "Люди проявляют ко мне столько сочувствия и симпатии, сколько я заслуживаю."},
        {"id": "stab_med4", "text": "Иногда мне в голову приходят такие нехорошие мысли, что лучше о них никому не рассказывать."},
        {"id": "stab_med5", "text": "Должен признать, что временами я волнуюсь из-за пустяков."},
        {"id": "stab_med6", "text": "Я часто предаюсь грустным размышлениям."},
        {"id": "stab_med7", "text": "Я человек нервный и легковозбудимый.."},
    ],
    "military_adaptation": [
        {"id": "mil_med1", "text": "Мне трудно принимать решения в стрессовых ситуациях."},
        {"id": "mil_med2", "text": "Я плохо переношу критику от старших."},
        {"id": "mil_med3", "text": "Мне сложно работать в команде."},
        {"id": "mil_med4", "text": "Я избегаю ответственности за других людей."},
        {"id": "mil_med5", "text": "Мне трудно соблюдать строгий распорядок дня."},
    ],
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
        {"id": "ag_full10", "text": "Я дерусь чаще, чем окружающие."},
    ],
    "isolation": [
        {"id": "is_high1", "text": "Я несчастлив, занимаясь столькими вещами в одиночку."},
        {"id": "is_high2", "text": "Я чувствую себя изолированным от других."},
        {"id": "is_high3", "text": "Я чувствую себя покинутым."},
        {"id": "is_high4", "text": "Я впечатлительнее большинства других людей."},
        {"id": "is_high5", "text": "я несчастен будучи таким отверженным."},
        {"id": "is_high6", "text": "Я чувствую себя совершенно одиноким."},
    ],
    "somatic": [
        {"id": "som_high1", "text": "Голова у меня болит часто."},
        {"id": "som_high2", "text": "Иногда мой слух настолько обостряется, что это мне даже мешает."},
        {"id": "som_high3", "text": "Иногда я чувствую, что у меня затрудненное дыхание"},
        {"id": "som_high4", "text": "Иногда я чувствую страх смерти"},
        {"id": "som_high5", "text": "Раз в неделю или чаще я бываю возбужденным и взволнованным."},
        {"id": "som_high6", "text": "Иногда я принимаю валериану, элениум или другие успокаивающие средства."},
    ],
    "anxiety": [
        {"id": "anx_high1", "text": "Я испытываю страх, кажется, будто что-то ужасное может вот-вот случиться"},
        {"id": "anx_high2", "text": "Некоторые вещи настолько меня волнуют, что мне даже говорить о них трудно."},
        {"id": "anx_high3", "text": "Иногда меня подводят нервы"},
        {"id": "anx_high4", "text": "Думаю, что я человек обреченный."},
        {"id": "anx_high5", "text": "Временами я бываю совершенно уверен в своей никчемности."},
    ],
    "stability": [
        {"id": "stab_high1", "text": "Теперь мне трудно надеяться на то, что я чего-нибудь добьюсь в жизни."},
        {"id": "stab_high2", "text": "Я легко теряю терпение с людьми."},
        {"id": "stab_high3", "text": "У меня бывали периоды, когда я что-то делал, а потом не знал, что именно я делал."},
        {"id": "stab_high4", "text": "Иногда у меня бывает чувство, что передо мной нагромоздилось столько трудностей, что одолеть их просто невозможно."},
        {"id": "stab_high5", "text": "Если в моих неудачах кто-то виноват, я не оставляю его безнаказанным."},
        {"id": "stab_high6", "text": "Мне очень трудно приспособиться к новым условиям жизни, работы или учебы. Переход к новым условиям жизни, работы или учебы кажется мне невыносимо трудным."},
        {"id": "stab_high7", "text": "Иногда я чувствую, что близок к нервному срыву."},
    ],
    "military_adaptation": [
        {"id": "mil_high1", "text": "Мне трудно принимать решения в стрессовых ситуациях."},
        {"id": "mil_high2", "text": "Я плохо переношу критику от старших."},
        {"id": "mil_high3", "text": "Мне сложно работать в команде."},
        {"id": "mil_high4", "text": "Я избегаю ответственности за других людей."},
        {"id": "mil_high5", "text": "Мне трудно соблюдать строгий распорядок дня."},
    ],
}

# Человекочитаемые названия шкал
SCALE_NAMES = {
    "aggression": "Шкала агрессии (Басса-Перри)",
    "isolation": "Шкала изоляции/депривации (Д. Рассел)",
    "somatic": "Шкала соматической депрессии (Бека)",
    "anxiety": "Шкала тревожности и депрессии (NUDS)",
    "stability": "Шкала нервно-психической устойчивости",
    "military_adaptation": "Шкала военной адаптации",
    "sincerity": "Шкала искренности",
}

# ---------------------------------------------------------------------------
# СТРУКТУРА ВОЕННОЙ АНКЕТЫ
# ---------------------------------------------------------------------------
#
# QUESTIONNAIRE_FIELDS — словарь section_key -> описание секции:
#   {
#     "title": <русское название секции>,
#     "fields": [ {"id", "label", "type", "section", ["options"], ["min"], ["max"], ["required"]} , ... ]
#   }
#
# Поле "label" дублирует исходный "text" из app.py (там анкета хранила вопрос в "text").
# Типы: text / textarea / date / select / radio / multiselect / number / slider.
#
# SECTIONS — упорядоченный список ключей секций (как в app.py MILITARY_QUESTIONNAIRE).

SECTIONS = [
    "personal_info",
    "achievements_family",
    "family_info",
    "social_connections",
    "health_history",
    "work_military",
    "religion_lifestyle",
    "financial_health",
]

QUESTIONNAIRE_FIELDS = {
    "personal_info": {
        "title": "👤 Личная информация",
        "fields": [
            {"id": "full_name", "label": "ФИО", "type": "text", "required": True, "section": "personal_info"},
            {"id": "birth_date", "label": "Дата рождения", "type": "date", "required": True, "section": "personal_info"},
            {"id": "birth_place", "label": "Место рождения", "type": "text", "required": True, "section": "personal_info"},
            {"id": "residence", "label": "Место жительства", "type": "text", "required": True, "section": "personal_info"},
            {"id": "residence_coliving", "label": "С кем в настоящее время проживаете и в течении какого времени", "type": "text", "required": True, "section": "personal_info"},
            {"id": "team_senior", "label": "Старший команды", "type": "text", "required": False, "section": "personal_info"},
            {"id": "nationality", "label": "Национальность", "type": "text", "required": True, "section": "personal_info"},
            {"id": "marital_status", "label": "Семейное положение", "type": "select", "options": ["Холост", "Женат", "Разведен"], "required": True, "section": "personal_info"},
            {"id": "education", "label": "Образование", "type": "select", "options": ["Среднее", "Среднее специальное", "Высшее", "Неполное высшее"], "required": True, "section": "personal_info"},
            {"id": "social_media", "label": "Укажите ваши аккаунты в соц сетях", "type": "textarea", "required": False, "section": "personal_info"},
        ],
    },
    "achievements_family": {
        "title": "🏆 Достижения и семья",
        "fields": [
            {"id": "sports_achievements", "label": "Есть ли у вас спортивные достижения? Какие?", "type": "textarea", "required": False, "section": "achievements_family"},
            {"id": "family_completeness", "label": "Вы воспитывались в полной/неполной семье", "type": "select", "options": ["Полной", "Неполной"], "required": True, "section": "achievements_family"},
            {"id": "deceased_relatives", "label": "Есть ли умершие среди близких родственников? (кто, год смерти, причина)", "type": "textarea", "required": False, "section": "achievements_family"},
        ],
    },
    "family_info": {
        "title": "👨‍👩‍👧‍👦 Информация о семье",
        "fields": [
            {"id": "father_info", "label": "ФИО отца, возраст, место работы", "type": "textarea", "required": False, "section": "family_info"},
            {"id": "father_relationship", "label": "Взаимоотношения с отцом", "type": "select", "options": ["Отличные", "Хорошие", "Удовлетворительные", "Плохие", "Отсутствуют"], "required": False, "section": "family_info"},
            {"id": "mother_info", "label": "ФИО матери, возраст, место работы", "type": "textarea", "required": False, "section": "family_info"},
            {"id": "mother_relationship", "label": "Взаимоотношения с матерью", "type": "select", "options": ["Отличные", "Хорошие", "Удовлетворительные", "Плохие", "Отсутствуют"], "required": False, "section": "family_info"},
            {"id": "siblings", "label": "Братья и сестры (ФИО, возраст)", "type": "textarea", "required": False, "section": "family_info"},
            {"id": "home_escapes", "label": "Бывали ли у вас случаи побегов из дома?", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "family_info"},
        ],
    },
    "social_connections": {
        "title": "🌐 Социальные связи",
        "fields": [
            {"id": "astana_contacts", "label": "Есть ли в городе Астана родственники или знакомые (ФИО и адрес)", "type": "textarea", "required": False, "section": "social_connections"},
            {"id": "family_suicides", "label": "Были ли самоубийства или суицидальные попытки у родственников", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "social_connections"},
            {"id": "personal_suicides", "label": "Имелись ли у вас в прошлом суицидальные попытки/мысли", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "social_connections"},
        ],
    },
    "health_history": {
        "title": "🏥 Медицинская история",
        "fields": [
            {"id": "family_alcoholism", "label": "Был ли в вашей семье алкоголизм", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "family_drugs", "label": "Была ли в вашей семье наркомания", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "family_criminal", "label": "Была ли в вашей семье судимость", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "family_mental", "label": "Были ли в семье наследственные нервно-психические заболевания", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_alcoholism", "label": "Были ли у вас до армии факты алкоголизма", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_drugs", "label": "Были ли у вас до армии факты наркомании", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_criminal", "label": "Были ли у вас до армии судимости", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_mental", "label": "Были ли у вас до армии нервно-психические заболевания", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_headtrauma", "label": "Были ли у Вас сотрясения мозга/травмы головы", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "personal_gambling", "label": "Была ли у вас игромания", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "hereditary_diseases", "label": "Имеешь ли ты тяжёлые наследственные заболевания? (онкологические, дыхательные, гипертония, сердечные и т.д.)", "type": "textarea", "required": False, "section": "health_history"},
            {"id": "seizures", "label": "Были ли у ближайших родственников или у вас судорожные припадки", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "bedwetting", "label": "Было ли у вас ночное недержание мочи?", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "health_history"},
            {"id": "bedwetting_age", "label": "В каком возрасте? (если было недержание)", "type": "number", "required": False, "section": "health_history"},
        ],
    },
    "work_military": {
        "title": "💼 Работа и военная служба",
        "fields": [
            {"id": "work_before_army", "label": "Кем работал до армии, сколько времени?", "type": "textarea", "required": False, "section": "work_military"},
            {"id": "want_serve", "label": "Желаете ли вы проходить военную службу", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "work_military"},
            {"id": "serve_reason", "label": "Причина (если не желаете служить)", "type": "textarea", "required": False, "section": "work_military"},
            {"id": "service_difficulties", "label": "В чем для вас будет трудность воинской службы", "type": "multiselect", "options": ["Беспрекословное подчинение", "Физические нагрузки", "Удаленность от дома", "Высокая личная ответственность", "Преодоление собственных отрицательных привычек", "Другое"], "required": True, "section": "work_military"},
        ],
    },
    "religion_lifestyle": {
        "title": "🕌 Религия и образ жизни",
        "fields": [
            {"id": "religion_type", "label": "Какую религию исповедуешь", "type": "text", "required": False, "section": "religion_lifestyle"},
            {"id": "religion_direction", "label": "Какое направление религии", "type": "text", "required": False, "section": "religion_lifestyle"},
            {"id": "religion_teachers", "label": "Если ты слушаешь духовных учителей, то перечисли их", "type": "text", "required": False, "section": "religion_lifestyle"},
            {"id": "religious_attendance", "label": "Как часто ходишь в мечеть/церковь", "type": "select", "options": ["Каждый день", "Несколько раз в неделю", "Раз в неделю", "Несколько раз в месяц", "Редко", "Никогда"], "required": False, "section": "religion_lifestyle"},
            {"id": "traditional_holidays", "label": "Празднуете ли вы традиционные праздники?", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "religion_lifestyle"},
            {"id": "social_events", "label": "Ходите на различные торжества (дни рождения, свадьбы)", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "religion_lifestyle"},
            {"id": "girlfriend", "label": "Есть ли девушка?", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "religion_lifestyle"},
            {"id": "relationship_closeness", "label": "Насколько близкие отношения по шкале от 1 до 5", "type": "slider", "min": 1, "max": 5, "required": False, "section": "religion_lifestyle"},
            {"id": "relationship_period", "label": "Сколько времени вы в отношениях", "type": "text", "required": False, "section": "religion_lifestyle"},
        ],
    },
    "financial_health": {
        "title": "💰 Финансы и здоровье",
        "fields": [
            {"id": "betting", "label": "Делаешь ли ставки в букмекерских конторах или онлайн", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "financial_health"},
            {"id": "credits", "label": "Есть у тебя кредиты/займы (сколько, на какую сумму, кто оплачивает)", "type": "textarea", "required": False, "section": "financial_health"},
            {"id": "medical_examination", "label": "При прохождении ВВК в ДДО полностью ли вы прошли обследование у врачей", "type": "radio", "options": ["Да", "Нет"], "required": True, "section": "financial_health"},
            {"id": "hidden_health_facts", "label": "Есть ли факты относительно вашего здоровья (диагнозы по которым ранее вас не брали на службу), о которых вы не сказали вашему старшему", "type": "textarea", "required": False, "section": "financial_health"},
        ],
    },
}


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------

def _question_belongs_to_scale(question_id, scale, bank):
    """True, если question_id присутствует в банке bank[scale]."""
    return any(q["id"] == question_id for q in bank.get(scale, []))


def _coerce_value(value):
    """Привести ответ к int (ответы теста — 1..5). Нечисловые -> 0."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# 1. ПЕРВИЧНЫЙ СКРИНИНГ
# ---------------------------------------------------------------------------

def analyze_screening(responses):
    """
    Анализ результатов первичного скрининга.

    Перенос логики из analyze_results() в app.py.

    responses: dict question_id -> value (1..5).

    Возвращает dict:
      {
        "scale_scores":      {scale: int сумма баллов},
        "positive_answers":  {scale: int кол-во ответов value>=4},
        "risk_levels":       {scale: "low"|"medium"|"warning"},
        "risk_levels_desc":  {scale: текстовое описание},
        "medium_risk_scales": [scale, ...],
        "high_risk_scales":   [],   # на этапе скрининга всегда пуст
      }
    """
    scores = {}
    positive_answers = {}

    # Расчёт баллов и положительных ответов по каждой шкале.
    for scale in SCREENING_QUESTIONS.keys():
        score = 0
        count = 0
        positive_count = 0

        for question_id, value in responses.items():
            # Определяем шкалу вопроса (первое совпадение — как в app.py).
            for s, questions in SCREENING_QUESTIONS.items():
                if any(q["id"] == question_id for q in questions):
                    if s == scale:
                        v = _coerce_value(value)
                        score += v
                        count += 1
                        if v >= 4:  # ответ считается положительным при 4 или 5
                            positive_count += 1
                    break

        if count > 0:
            scores[scale] = score
            positive_answers[scale] = positive_count

    # Определение уровней риска.
    risk_levels = {}
    risk_levels_desc = {}
    medium_risk_scales = []
    high_risk_scales = []  # на скрининге не заполняется

    for scale, score in scores.items():
        if scale == "sincerity":
            # Спец. логика для шкалы искренности.
            if score >= 13 or score <= 4:
                risk_levels[scale] = "warning"
                risk_levels_desc[scale] = "низкая искренность ответов"
            # NB: если 5 <= score <= 12, app.py не записывает sincerity
            #     ни в risk_levels, ни в risk_levels_desc — поведение сохранено.
        else:
            if positive_answers.get(scale, 0) >= 2:
                risk_levels[scale] = "medium"
                risk_levels_desc[scale] = "средний уровень риска (требуется дополнительная оценка)"
                medium_risk_scales.append(scale)
            else:
                risk_levels[scale] = "low"
                risk_levels_desc[scale] = "низкий уровень риска"

    return {
        "scale_scores": scores,
        "positive_answers": positive_answers,
        "risk_levels": risk_levels,
        "risk_levels_desc": risk_levels_desc,
        "medium_risk_scales": medium_risk_scales,
        "high_risk_scales": high_risk_scales,
    }


# ---------------------------------------------------------------------------
# 2. УГЛУБЛЁННАЯ ОЦЕНКА ПО ШКАЛЕ
# ---------------------------------------------------------------------------

def evaluate_detailed(scale, risk_level, responses):
    """
    Подсчёт результатов углублённой оценки по одной шкале.

    Перенос вычислительной части complete_detailed_assessment() из app.py
    (без управления переходами по этапам — только скоринг и решение об эскалации).

    scale:      ключ шкалы.
    risk_level: "medium" или "high" — какой банк вопросов использовать.
    responses:  dict question_id -> value (1..5); может содержать ответы и других
                этапов — фильтруются по соответствующему банку.

    Возвращает dict:
      {
        "score":            int,
        "max_possible":     int (total_questions * 5),
        "percentage":       float (0..100),
        "positive_count":   int (value>=4),
        "total_questions":  int,
        "new_risk_desc":    str | None — новое описание уровня риска шкалы,
        "escalate_to_high": bool — нужно ли перейти к высокой оценке
                            (только для medium при >50% положительных),
      }

    Если по банку нет ни одного отвеченного вопроса (total_questions == 0),
    возвращается нулевой результат с new_risk_desc=None, escalate_to_high=False.
    """
    if risk_level == "medium":
        bank = MEDIUM_RISK_QUESTIONS
    elif risk_level == "high":
        bank = HIGH_RISK_QUESTIONS
    else:
        bank = {}

    detailed_score = 0
    question_count = 0
    positive_count = 0

    for question_id, value in responses.items():
        if _question_belongs_to_scale(question_id, scale, bank):
            v = _coerce_value(value)
            detailed_score += v
            question_count += 1
            if v >= 4:
                positive_count += 1

    if question_count == 0:
        return {
            "score": 0,
            "max_possible": 0,
            "percentage": 0.0,
            "positive_count": 0,
            "total_questions": 0,
            "new_risk_desc": None,
            "escalate_to_high": False,
        }

    max_possible = question_count * 5
    percentage = (detailed_score / max_possible) * 100

    new_risk_desc = None
    escalate_to_high = False

    if risk_level == "medium":
        # Более 50% положительных -> эскалация в high.
        if positive_count > question_count / 2:
            new_risk_desc = "высокий уровень риска (подтверждено углубленной оценкой)"
            escalate_to_high = True
        else:
            new_risk_desc = "средний уровень риска (подтверждено углубленной оценкой)"
    elif risk_level == "high":
        if percentage >= 70:
            new_risk_desc = "высокий уровень риска (подтверждено углубленной оценкой)"
        else:
            new_risk_desc = "средний уровень риска (скорректировано после углубленной оценки)"

    return {
        "score": detailed_score,
        "max_possible": max_possible,
        "percentage": percentage,
        "positive_count": positive_count,
        "total_questions": question_count,
        "new_risk_desc": new_risk_desc,
        "escalate_to_high": escalate_to_high,
    }


# ---------------------------------------------------------------------------
# 3. ИТОГОВАЯ РЕКОМЕНДАЦИЯ
# ---------------------------------------------------------------------------

def final_recommendation(questionnaire, risk_levels_desc):
    """
    Формирование итоговой рекомендации.

    Перенос логики prepare_final_recommendations() из app.py.
    Debug-вывод не переносится (он уже удалён в исходнике).

    questionnaire:    dict ответов военной анкеты (question_id -> value).
    risk_levels_desc: dict scale -> текстовое описание уровня риска
                      (как из analyze_screening / evaluate_detailed).

    Возвращает dict:
      {
        "recommendation": "recommended" | "recommended_with_restrictions" | "not_recommended",
        "messages": [str, ...],  # человекочитаемые строки-обоснования
      }

    Правила (в точности как в app.py):
      - Высокий риск по любой шкале (в описании есть "высокий") -> not_recommended.
      - Риск экстремизма (религ. учитель указан + частое посещение
        ['Несколько раз в неделю','Каждый день'] + social_events == 'Нет')
        -> not_recommended.
      - Иначе при наличии шкал среднего риска (в описании есть "средний")
        -> recommended_with_restrictions.
      - Иначе -> recommended.
    """
    messages = []
    questionnaire = questionnaire or {}
    risk_levels_desc = risk_levels_desc or {}

    has_high_risk = False
    for scale, level in risk_levels_desc.items():
        if "высокий" in level:
            has_high_risk = True
            messages.append(
                "⚠️ Выявлен высокий уровень риска по шкале '%s'"
                % SCALE_NAMES.get(scale, scale)
            )

    # Проверка риска экстремизма.
    has_religious_teacher = bool(str(questionnaire.get("religion_teachers", "")).strip())
    frequent_attendance = questionnaire.get("religious_attendance") in [
        "Несколько раз в неделю",
        "Каждый день",
    ]
    no_social_events = questionnaire.get("social_events") == "Нет"

    if has_religious_teacher and frequent_attendance and no_social_events:
        has_high_risk = True
        messages.append("⚠️ Выявлен риск экстремизма")

    if has_high_risk:
        messages.append("❌ **Не рекомендуется к военной службе**")
        recommendation = "not_recommended"
    else:
        medium_risk_scales = []
        for scale, level in risk_levels_desc.items():
            if "средний" in level:
                medium_risk_scales.append(SCALE_NAMES.get(scale, scale))

        if medium_risk_scales:
            messages.append(
                "⚠️ Требуется дополнительное внимание к следующим аспектам: %s"
                % ", ".join(medium_risk_scales)
            )
            messages.append("✅ **Рекомендуется к военной службе с ограничениями**")
            recommendation = "recommended_with_restrictions"
        else:
            messages.append("✅ **Рекомендуется к военной службе**")
            recommendation = "recommended"

    return {
        "recommendation": recommendation,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# МИНИ-САМОПРОВЕРКА
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Пример: два положительных ответа (value=5) по шкале агрессии ->
    # ожидаем medium-риск по 'aggression'.
    sample_responses = {
        "ag1": 5,
        "ag2": 5,
        "ag3": 2,
        "is1": 1,
        "is2": 1,
        "is3": 1,
        "sin1": 1,
        "sin2": 1,
        "sin3": 1,
    }

    print("=== analyze_screening ===")
    screening = analyze_screening(sample_responses)
    for key, val in screening.items():
        print("%-20s %s" % (key + ":", val))

    print("\n=== evaluate_detailed (aggression / medium) ===")
    detail_responses = {
        "ag_med1": 5,
        "ag_med2": 5,
        "ag_med3": 5,
        "ag_med4": 2,
        "ag_med5": 1,
    }
    detailed = evaluate_detailed("aggression", "medium", detail_responses)
    for key, val in detailed.items():
        print("%-20s %s" % (key + ":", val))

    print("\n=== final_recommendation ===")
    rec = final_recommendation(
        questionnaire={
            "religion_teachers": "",
            "religious_attendance": "Редко",
            "social_events": "Да",
        },
        risk_levels_desc=screening["risk_levels_desc"],
    )
    print("recommendation:", rec["recommendation"])
    for msg in rec["messages"]:
        print(" -", msg)

    print("\nOK: module imported and ran self-check successfully.")
