@echo off
CHCP 65001 >nul
java -cp program/app/target/open-all-gpt.jar huplay.AppNetworkWorker %*