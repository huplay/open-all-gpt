java -Xmx7168m -Xms7168m -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=8888 ^
    -cp d:/Java/huplay/open-all-gpt/target/open-all-gpt.jar ^
    app.AppStandaloneMain "EleutherAI/GPT-J/GPT-J 6B" -max=25 -topK=40