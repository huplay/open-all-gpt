@echo off
CHCP 65001 >nul
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=8002
     -cp target/open-all-gpt.jar app.AppNetworkClient %*
