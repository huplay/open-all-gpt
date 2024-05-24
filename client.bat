@echo off
CHCP 65001 >nul
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=8002
     -cp program/app/target/open-all-gpt.jar huplay.AppNetworkClient %*
