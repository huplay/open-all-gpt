java -Xmx7168m -Xms7168m -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=8888 ^
    -cp d:/Java/huplay/open-all-gpt/app/target/open-all-gpt.jar ^
    app.AppStandaloneMain "Google/Gemma/Gemma 2B" -max=25 -topK=40