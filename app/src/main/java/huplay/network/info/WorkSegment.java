package huplay.network.info;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import huplay.network.Address;

import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class WorkSegment
{
    private Address worker;

    private WorkSegmentType workSegmentType;
    private final List<DecoderBlock> decoderBlocks = new ArrayList<>();

    public WorkSegment() {} // Empty constructor for deserialization

    public WorkSegment(Address worker)
    {
        this.worker = worker;
    }

    public void addDecoderBlock(DecoderBlock decoderBlock)
    {
        decoderBlocks.add(decoderBlock);
    }

    public void setWorkSegmentType(WorkSegmentType workSegmentType)
    {
        this.workSegmentType = workSegmentType;
    }

    // Getters
    public Address getWorker() {return worker;}
    public WorkSegmentType getWorkSegmentType() {return workSegmentType;}
    public List<DecoderBlock> getDecoderBlocks() {return decoderBlocks;}

    @Override
    public String toString()
    {
        return "WorkSegment{" +
                "worker=" + worker +
                ", workSegmentType=" + workSegmentType +
                ", decoderBlocks=" + decoderBlocks +
                '}';
    }
}
