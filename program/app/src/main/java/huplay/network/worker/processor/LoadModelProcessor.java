package huplay.network.worker.processor;

import huplay.network.Address;
import huplay.network.message.Acknowledge;
import huplay.network.message.toWorker.LoadModelMessage;
import huplay.network.worker.task.LoadModelTask;

public class LoadModelProcessor
{
    public static Acknowledge process(LoadModelMessage message, Address serverAddress)
    {
        var modelId = message.getModelId();

        System.out.println("LoadModelRequest received: " + modelId + ", " + message.getWorkSegment());

        var task = new LoadModelTask(message, serverAddress);
        new Thread(task).start();

        return new Acknowledge();
    }
}
