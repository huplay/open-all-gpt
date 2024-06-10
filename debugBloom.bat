java -Xmx4096m -Xms4096m -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=8888 ^
    -cp d:/Java/huplay/open-all-gpt/app/target/open-all-gpt.jar ^
    app.AppStandaloneMain "BigScience/BLOOM/BLOOM 7.1B - GPTQ-4bit" -max=25 -topK=40