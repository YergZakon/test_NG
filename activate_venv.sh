#!/bin/bash

echo "Активация виртуального окружения..."
source venv/bin/activate
echo "Виртуальное окружение активировано. Вы можете запустить приложение командой:"
echo "streamlit run streamlit-app.py"
echo ""
exec $SHELL 