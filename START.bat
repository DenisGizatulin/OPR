@echo off
rem Запуск бэкенда
start cmd /k "cd %~dp0backend && uvicorn main:app --reload"

rem Запуск фронтенда
start cmd /k "cd %~dp0frontend && npm start"
