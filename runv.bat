@echo off
CHCP 65001 >nul
java --add-modules=jdk.incubator.vector -cp program/app/target/open-all-gpt.jar huplay.AppStandaloneLauncher %*
