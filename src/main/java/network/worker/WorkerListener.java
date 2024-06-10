package network.worker;

import network.Address;
import network.BaseHttpServer;
import network.message.BaseMessage;
import network.message.toWorker.WorkMessage;
import network.message.toWorker.LoadModelMessage;
import network.worker.processor.LoadModelProcessor;
import network.worker.processor.WorkRequestProcessor;

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
