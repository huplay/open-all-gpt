package network.message.toWorker;

import config.ModelConfig;
import network.message.Acknowledge;
import network.message.BaseRequest;
import network.info.WorkSegment;

public class LoadModelMessage extends BaseRequest<Acknowledge>
{
    private String taskUUID;
    private String modelId;
    private ModelConfig modelConfig;
    private WorkSegment workSegment;

    public LoadModelMessage() {} // Empty constructor for deserialization

    public LoadModelMessage(String taskUUID, String modelId, ModelConfig modelConfig,
                            WorkSegment workSegment)
    {
        this.taskUUID = taskUUID;
        this.modelId = modelId;
        this.modelConfig = modelConfig;
        this.workSegment = workSegment;
    }

    // Getters
    public String getTaskUUID() {return taskUUID;}
    public String getModelId() {return modelId;}
    public ModelConfig getModelConfig() {return modelConfig;}
    public WorkSegment getWorkSegment() {return workSegment;}
}
