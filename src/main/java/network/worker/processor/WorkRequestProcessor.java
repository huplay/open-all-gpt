package network.worker.processor;

import network.Address;
import network.message.Acknowledge;
import network.message.toWorker.WorkMessage;
import network.worker.task.WorkExecutionTask;

public class WorkRequestProcessor
{
    public static Acknowledge process(WorkMessage message, Address serverAddress)
    {
        String modelId = message.getModelId();
        System.out.println("WorkRequest received: " + modelId + ", workSegment: " + message.getWorkSegment());

        WorkExecutionTask task = new WorkExecutionTask(message, serverAddress);
        new Thread(task).start();

        return new Acknowledge();
    }
}
