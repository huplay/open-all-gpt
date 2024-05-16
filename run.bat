@echo off
CHCP 65001 >nul
java -cp app/target/open-all-gpt.jar huplay.AppStandaloneLauncher %*
