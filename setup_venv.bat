@echo off
echo ============================================================
echo ВНИМАНИЕ: Этот скрипт должен запускаться через cmd.exe
echo Если вы используете PowerShell и получаете ошибку выполнения,
echo запустите cmd.exe и выполните этот скрипт в нём.
echo ============================================================
echo.
echo Создание виртуального окружения...
python -m venv venv
echo Активация виртуального окружения...
call venv\Scripts\activate.bat
echo Установка зависимостей...
pip install -r requirements.txt
echo Создание примера файла .env...
if not exist .env (
    echo # Укажите ваш ключ API OpenAI> .env
    echo OPENAI_API_KEY=ваш_ключ_api>> .env
)
echo.
echo Настройка завершена! Чтобы запустить приложение, выполните:
echo   cmd.exe
echo   cd %cd%
echo   venv\Scripts\activate.bat
echo   streamlit run streamlit-app.py
echo.
pause 