package network.worker.processor;

import network.Address;
import network.message.Acknowledge;
import network.message.toWorker.LoadModelMessage;
import network.worker.task.LoadModelTask;

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
