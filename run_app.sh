#!/bin/bash

echo "Активация виртуального окружения..."
source venv/bin/activate
echo "Запуск приложения..."
streamlit run streamlit-app.py 