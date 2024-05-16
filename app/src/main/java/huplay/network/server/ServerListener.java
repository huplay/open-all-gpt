package huplay.network.server;

import huplay.network.BaseHttpServer;
import huplay.network.message.BaseMessage;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;
import huplay.network.message.toServer.fromClient.PollOpenModel;
import huplay.network.message.toServer.fromClient.PollQueryResult;
import huplay.network.message.toServer.fromClient.QueryRequest;
import huplay.network.message.toServer.fromClient.StartSessionRequest;
import huplay.network.server.processor.*;

public class ServerListener extends BaseHttpServer
{
    @Override
    protected BaseMessage handlePostMessage(BaseMessage received)
    {
        return switch (received)
        {
            case WorkerJoinedMessage message    -> WorkerJoinedProcessor.process(message);
            case PollOpenModel message          -> PollOpenModelProcessor.process(message);
            case ModelLoadedMessage message     -> ModelLoadedProcessor.process(message);
            case StartSessionRequest ignored    -> StartSessionProcessor.process();
            case QueryRequest message           -> QueryRequestProcessor.process(message);
            case WorkResultMessage message      -> WorkResultProcessor.process(message);
            case PollQueryResult message        -> PollQueryResultProcessor.process(message);
            default ->
                    throw new IllegalStateException("Unexpected value: " + received);
        };
    }

    @Override
    protected String handleGetMessage(String message)
    {
        return "<html><body>Open All GPT</body></html>";
    }
}
