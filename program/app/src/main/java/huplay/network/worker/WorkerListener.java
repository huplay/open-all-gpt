package huplay.network.worker;

import huplay.network.Address;
import huplay.network.BaseHttpServer;
import huplay.network.message.BaseMessage;
import huplay.network.message.toWorker.WorkMessage;
import huplay.network.message.toWorker.LoadModelMessage;
import huplay.network.worker.processor.LoadModelProcessor;
import huplay.network.worker.processor.WorkRequestProcessor;

public class WorkerListener extends BaseHttpServer
{
    private final Address serverAddress;

    public WorkerListener(Address serverAddress)
    {
        this.serverAddress = serverAddress;
    }

    @Override
    protected BaseMessage handlePostMessage(BaseMessage received)
    {
        return switch (received)
        {
            case LoadModelMessage message   -> LoadModelProcessor.process(message, serverAddress);
            case WorkMessage message        -> WorkRequestProcessor.process(message, serverAddress);
            default ->
                    throw new IllegalStateException("Unexpected value: " + received);
        };
    }

    @Override
    protected String handleGetMessage(String context, String path, String query)
    {
        return "<html><body>Open All GPT</body></html>";
    }
}
