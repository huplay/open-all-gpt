package huplay.network.message.toServer.fromWorker;

import huplay.network.Address;
import huplay.network.message.Acknowledge;
import huplay.network.message.BaseRequest;

public class WorkerJoinedMessage extends BaseRequest<Acknowledge>
{
    private Address workerAddress;
    private long freeMemory;

    public WorkerJoinedMessage() {} // Empty constructor for deserialization

    public WorkerJoinedMessage(Address sender, long freeMemory)
    {
        this.workerAddress = sender;
        this.freeMemory = freeMemory;
    }

    public long getFreeMemory() {return freeMemory;}
    public Address getWorkerAddress() {return workerAddress;}
}
