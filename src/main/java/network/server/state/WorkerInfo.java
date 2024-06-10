package network.server.state;

import network.Address;

public class WorkerInfo implements Comparable<WorkerInfo>
{
    private final Address address;
    private long freeMemory;

    public WorkerInfo(Address address, long freeMemory)
    {
        this.address = address;
        this.freeMemory = freeMemory;
    }

    // Getters, setters
    public Address getAddress() {return address;}
    public long getFreeMemory() {return freeMemory;}
    public void setFreeMemory(long freeMemory) {this.freeMemory = freeMemory;}

    @Override
    public int compareTo(WorkerInfo that)
    {
        return Long.compare(this.freeMemory, that.freeMemory);
    }
}
