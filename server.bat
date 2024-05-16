@echo off
CHCP 65001 >nul
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=8000 -cp app/target/open-all-gpt.jar huplay.AppNetworkServer %*