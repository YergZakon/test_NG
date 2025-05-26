#!/bin/bash

echo "Создание виртуального окружения..."
python3 -m venv venv
echo "Активация виртуального окружения..."
source venv/bin/activate
echo "Установка зависимостей..."
pip install -r requirements.txt
echo "Создание примера файла .env..."
if [ ! -f .env ]; then
    echo "# Укажите ваш ключ API OpenAI" > .env
    echo "OPENAI_API_KEY=ваш_ключ_api" >> .env
fi

echo ""
echo "Настройка завершена! Чтобы запустить приложение, выполните:"
echo "  source venv/bin/activate"
echo "  streamlit run streamlit-app.py"
echo "" 