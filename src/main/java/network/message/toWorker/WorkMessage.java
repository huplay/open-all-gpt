package network.message.toWorker;

import network.message.Acknowledge;
import network.message.BaseRequest;
import network.info.input.Input;
import network.info.WorkSegment;
import network.server.state.QueryState;

public class WorkMessage extends BaseRequest<Acknowledge>
{
    private String workUUID;
    private String modelId;
    private Input input;
    private int topK;
    private boolean inputOnly;
    private WorkSegment workSegment;

    public WorkMessage() {} // Empty constructor for deserialization

    public WorkMessage(String workUUID, QueryState queryState, Input input, WorkSegment workSegment)
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
