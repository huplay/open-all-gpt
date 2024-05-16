package huplay.network.server.processor;

import huplay.network.info.input.TokenInput;
import huplay.network.message.toServer.fromClient.QueryRequest;
import huplay.network.message.toServer.fromClient.QueryResponse;
import huplay.network.message.toWorker.WorkRequest;
import huplay.network.server.state.QueryState;
import huplay.tokenizer.Token;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import static huplay.network.server.state.ServerState.getServerState;

public class QueryRequestProcessor
{
    public static QueryResponse process(QueryRequest message)
    {
        // TODO: Make sure there won't be a queryRequest while the previous is in progress

        var serverState = getServerState();
        var modelId = message.getModelId();
        var sessionUUID = message.getSessionUUID();
        var inputText = message.getText();
        var topK = message.getTopK();
        var maxLength = message.getMaxLength();

        System.out.println("QueryRequest received for " + modelId);

        // Split the text to tokens
        var tokenizer = serverState.getTokenizer(modelId);

        List<Token> inputTokens = new ArrayList<>();
        if (inputText == null || inputText.isEmpty())
        {
            inputTokens.add(tokenizer.getEndOfTextToken());
        }
        else
        {
            inputTokens.addAll(tokenizer.split(inputText));
        }

        // Create new QueryState
        var queryUUID = UUID.randomUUID().toString();
        var workSegments = serverState.getWorkSegments(modelId);
        var queryState = new QueryState(modelId, sessionUUID, queryUUID, inputTokens, topK, maxLength, workSegments);
        serverState.addPendingQuery(queryUUID, queryState);

        // Create new pending work
        var workUUID = UUID.randomUUID().toString();
        serverState.addPendingWork(workUUID, queryState);

        // Create WorkRequest and send to the first worker in the flow
        var inputToken = inputTokens.getFirst().getId();
        var tokenInput = new TokenInput(0, inputToken);
        var workSegment = queryState.getActualWorkSegment();
        var workRequest = new WorkRequest(workUUID, queryState, tokenInput, workSegment);

        try
        {
            workRequest.send(workSegment.getWorker());
        }
        catch (IOException e)
        {
            System.out.println("ERROR sending the work request: " + workRequest);
        }

        return new QueryResponse(queryUUID, inputTokens);
    }
}
