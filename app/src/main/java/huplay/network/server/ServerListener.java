package huplay.network.server;

import huplay.network.BaseHttpServer;
import huplay.network.message.BaseMessage;
import huplay.network.message.toServer.fromClient.*;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;
import huplay.network.server.processor.*;
import huplay.network.server.servlet.ErrorServlet;
import huplay.network.server.servlet.MainServlet;


public class ServerListener extends BaseHttpServer
{
    @Override
    protected BaseMessage handlePostMessage(BaseMessage received)
    {
        return switch (received)
        {
            // From client:
            case ClientJoinedRequest ignored    -> ClientJoinedProcessor.process();
            case PollOpenModelRequest message          -> PollOpenModelProcessor.process(message);
            case StartSessionRequest ignored    -> StartSessionProcessor.process();
            case QueryRequest message           -> QueryRequestProcessor.process(message);
            case PollQueryResultRequest message        -> PollQueryResultProcessor.process(message);

            // From worker:
            case WorkerJoinedMessage message    -> WorkerJoinedProcessor.process(message);
            case ModelLoadedMessage message     -> ModelLoadedProcessor.process(message);
            case WorkResultMessage message      -> WorkResultProcessor.process(message);

            default ->
                    throw new IllegalStateException("Unexpected value: " + received);
        };
    }

    @Override
    protected String handleGetMessage(String context, String path, String query)
    {
        return switch (context)
        {
            case "/"            -> MainServlet.get(path, query);

            default -> ErrorServlet.get(path);
        };
    }
}
