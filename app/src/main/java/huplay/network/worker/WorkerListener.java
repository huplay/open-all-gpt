package huplay.network.worker;

import huplay.network.Address;
import huplay.network.BaseHttpServer;
import huplay.network.message.BaseMessage;
import huplay.network.message.toWorker.WorkRequest;
import huplay.network.message.toWorker.LoadModelRequest;
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
            case LoadModelRequest message   -> LoadModelProcessor.process(message, serverAddress);
            case WorkRequest message        -> WorkRequestProcessor.process(message, serverAddress);
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
