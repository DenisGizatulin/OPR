@echo off
rem Запуск бэкенда в скрытом окне
start /B cmd /c "cd %~dp0backend && uvicorn main:app --reload"

rem Запуск фронтенда в скрытом окне
start /B cmd /c "cd %~dp0frontend && npm start"

rem Закрытие текущего окна
exit