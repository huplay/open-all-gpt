java -Xmx4096m -Xms4096m -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=8888 ^
    -cp d:/Java/huplay/open-all-gpt/program/app/target/open-all-gpt.jar ^
    huplay.AppStandaloneMain "OpenAI/GPT 2/GPT 2 - QLoRA-4bit" -max=25 -topK=40