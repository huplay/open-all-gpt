java -Xmx8192m -Xms8192m -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=8888 ^
    -cp d:/Java/huplay/open-all-gpt/target/open-all-gpt.jar ^
    app.AppStandaloneMain "Google/Gemma/Gemma 2B" -max=25 -topK=40