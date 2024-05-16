package huplay.network.message.toWorker;

import huplay.config.ModelConfig;
import huplay.network.message.Acknowledge;
import huplay.network.message.BaseRequest;
import huplay.network.info.WorkSegment;

public class LoadModelRequest extends BaseRequest<Acknowledge>
{
    private String taskUUID;
    private String modelId;
    private ModelConfig modelConfig;
    private WorkSegment workSegment;

    public LoadModelRequest() {} // Empty constructor for deserialization

    public LoadModelRequest(String taskUUID, String modelId, ModelConfig modelConfig,
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
