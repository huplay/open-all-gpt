package huplay.network.server.processor;

import huplay.network.message.toServer.fromClient.PollQueryResult;
import huplay.network.message.toServer.fromClient.PollQueryResultResponse;

import static huplay.network.server.state.ServerState.getServerState;

public class PollQueryResultProcessor
{
    public static PollQueryResultResponse process(PollQueryResult message)
    {
        var modelId = message.getModelId();
        if (message.getAttempt() == 0) System.out.println("GetQueryResultRequest received for: " + modelId);

        var isReady = true;

        var serverState = getServerState();
        var queryUUID = message.getQueryUUID();
        var queryState = serverState.getFinishedQueries().get(queryUUID);
        if (queryState == null)
        {
            queryState = serverState.getPendingQueries().get(queryUUID);
            isReady = false;
        }

        var generatedTokens = queryState.getGeneratedTokens();
        var generatedText = queryState.getGeneratedText();
        return new PollQueryResultResponse(queryUUID, generatedTokens, generatedText, isReady);
    }
}
