package huplay.network.message.toServer.fromWorker;

import huplay.network.message.Acknowledge;
import huplay.network.message.BaseRequest;

public class ModelLoadedMessage extends BaseRequest<Acknowledge>
{
    private String taskUUID;
    private String modelId;

    public ModelLoadedMessage() {} // Empty constructor for deserialization

    public ModelLoadedMessage(String taskUUID, String modelId)
    {
        this.taskUUID = taskUUID;
        this.modelId = modelId;
    }

    // Getters
    public String getTaskUUID() {return taskUUID;}
    public String getModelId() {return modelId;}
}
