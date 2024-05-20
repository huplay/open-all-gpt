@echo off
CHCP 65001 >nul

start cmd /c server.bat 1234

start cmd /c worker.bat -join=localhost:1234 -port=9991
REM start cmd /c worker.bat -join=localhost:80 -port=9999

REM start cmd /c client.bat -join=localhost:80 -port=9990