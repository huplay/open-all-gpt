@echo off
CHCP 65001 >nul
java -cp target/open-all-gpt.jar app.AppStandaloneLauncher %*
