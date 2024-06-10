package network.message.toWorker;

import network.message.Acknowledge;
import network.message.BaseRequest;
import network.info.output.Output;
import network.info.WorkSegment;

public class WorkResultMessage extends BaseRequest<Acknowledge>
{
    private String workUUID;
    private WorkSegment workSegment;
    private Output result;

    public WorkResultMessage() {} // Empty constructor for deserialization

    public WorkResultMessage(String workUUID, WorkSegment workSegment, Output result)
    {
        this.workUUID = workUUID;
        this.workSegment = workSegment;
        this.result = result;
    }

    // Getters
    public String getWorkUUID() {return workUUID;}
    public WorkSegment getWorkSegment() {return workSegment;}
    public Output getResult() {return result;}
}
