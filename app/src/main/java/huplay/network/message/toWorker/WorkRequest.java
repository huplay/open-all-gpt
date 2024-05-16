package huplay.network.message.toWorker;

import huplay.network.message.Acknowledge;
import huplay.network.message.BaseRequest;
import huplay.network.info.input.Input;
import huplay.network.info.WorkSegment;
import huplay.network.server.state.QueryState;

public class WorkRequest extends BaseRequest<Acknowledge>
{
    private String workUUID;
    private String modelId;
    private Input input;
    private int topK;
    private boolean inputOnly;
    private WorkSegment workSegment;

    public WorkRequest() {} // Empty constructor for deserialization

    public WorkRequest(String workUUID, QueryState queryState, Input input, WorkSegment workSegment)
    {
        this.workUUID = workUUID;
        this.modelId = queryState.getModelId();
        this.input = input;
        this.topK = queryState.getTopK();
        this.inputOnly = queryState.isInputOnly();
        this.workSegment = workSegment;
    }

    // Getters
    public String getWorkUUID() {return workUUID;}
    public String getModelId() {return modelId;}
    public Input getInput() {return input;}
    public int getTopK() {return topK;}
    public boolean getInputOnly() {return inputOnly;}
    public WorkSegment getWorkSegment() {return workSegment;}
}
