package huplay.network.worker.processor;

import huplay.network.Address;
import huplay.network.message.Acknowledge;
import huplay.network.message.toWorker.WorkMessage;
import huplay.network.worker.task.WorkExecutionTask;

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
