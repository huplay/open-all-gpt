package huplay.network.server.processor;

import huplay.network.info.input.HiddenStateInput;
import huplay.network.info.input.TokenInput;
import huplay.network.info.output.EmptyOutput;
import huplay.network.info.output.HiddenStateOutput;
import huplay.network.info.output.Output;
import huplay.network.info.output.TokenOutput;
import huplay.network.message.Acknowledge;
import huplay.network.message.toWorker.WorkMessage;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.network.server.state.QueryState;
import huplay.tokenizer.Token;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import static huplay.network.info.WorkSegmentType.TAIL_ONLY;
import static huplay.network.server.state.ServerState.getServerState;

public class WorkResultProcessor
{
    public static Acknowledge process(WorkResultMessage message)
    {
        var serverState = getServerState();
        var workUUID = message.getWorkUUID();
        System.out.println("WorkResult received for " + workUUID);

        // Find the corresponding queryState and remove the work from the pending works
        var queryState = serverState.getPendingQueryState(workUUID);
        serverState.removePendingWork(workUUID);

        if (hasNextWorkSegment(queryState))
        {
            // If there is remaining work segment, send the request to the worker
            requestNextWorkSegmentProcess(queryState, message);
        }
        else
        {
            // All work segment was executed (one token process is finished), process the result
            int nextToken;
            var isFinished = false;

            Output result = message.getResult();
            if (result instanceof EmptyOutput)
            {
                // This is the result if it was an inputOnly processing. (Input token processing)
                queryState.incrementProcessedInput();
                nextToken = queryState.getActualInputToken();
            }
            else if (result instanceof TokenOutput tokenOutput)
            {
                // This is the result at the last input token, or at generated tokens
                nextToken = tokenOutput.getToken();
                isFinished = recordToken(queryState, nextToken);
            }
            else
            {
                throw new RuntimeException("Incorrect output type in the work result: " + result);
            }

            if (isFinished)
            {
                // We processed all tokens of the query
               serverState.activateQuery(queryState.getQueryUUID());
            }
            else
            {
                processNextToken(queryState, nextToken);
            }
        }

        return new Acknowledge();
    }

    private static boolean hasNextWorkSegment(QueryState queryState)
    {
        var workSegments = queryState.getWorkSegments();
        var segmentIndex = queryState.getWorkSegmentIndex();

        if (workSegments.size() <= segmentIndex + 1)
        {
            return false;
        }

        var nextSegmentType = workSegments.get(segmentIndex + 1).getWorkSegmentType();

        // If only the tail is remained, and we are not interested in the result, we can skip the last workSegment
        return ! (nextSegmentType.equals(TAIL_ONLY) && queryState.isInputOnly());
    }

    private static void requestNextWorkSegmentProcess(QueryState queryState, WorkResultMessage message)
    {
        // If we have a next workSegment, it is certain the previous output was a hiddenState
        var hiddenStateOutput = (HiddenStateOutput) message.getResult();
        var input = new HiddenStateInput(hiddenStateOutput.getHiddenState());

        var workSegment = queryState.nextWorkSegment();

        // Send a new work request to process the next workSegment
        var workUUID = UUID.randomUUID().toString();
        getServerState().addPendingWork(workUUID, queryState);

        var workRequest = new WorkMessage(workUUID, queryState, input, workSegment);

        try
        {
            workRequest.send(workSegment.getWorker());
        }
        catch (IOException e)
        {
            System.out.println("ERROR sending the work request: " + workRequest);
        }
    }

    private static boolean recordToken(QueryState queryState, int token)
    {
        // Encode the result by the tokenizer
        var tokenizer = getServerState().getTokenizer(queryState.getModelId());
        var tokenText = tokenizer.decode(List.of(token));
        queryState.getGeneratedTokens().add(new Token(token, tokenText));
        queryState.setGeneratedText(tokenizer.decode(queryState.getGeneratedTokenIds()));

        // Determine should we continue the generation
        var EOS = tokenizer.getEndOfTextToken().getId();

        return token == EOS || queryState.isMaxLengthReached();
    }

    private static void processNextToken(QueryState queryState, int nextToken)
    {
        // Create new pending work
        var serverState = getServerState();
        var workUUID = UUID.randomUUID().toString();
        serverState.addPendingWork(workUUID, queryState);

        // Create WorkRequest and send to the first worker in the flow
        queryState.resetWorkSegmentIndex();
        var workSegment = queryState.getActualWorkSegment();

        var pos = queryState.getPos();
        var tokenInput = new TokenInput(pos, nextToken);
        var workRequest = new WorkMessage(workUUID, queryState, tokenInput, workSegment);

        try
        {
            workRequest.send(workSegment.getWorker());
        }
        catch (IOException e)
        {
            System.out.println("ERROR sending the work request: " + workRequest);
        }
    }
}
