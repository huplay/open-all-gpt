@echo off
CHCP 65001 >nul
java --add-modules=jdk.incubator.vector -cp target/open-all-gpt.jar app.AppStandaloneLauncher %*
