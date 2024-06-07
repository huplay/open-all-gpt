package huplay.network.info;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class DecoderBlock
{
    private DecoderBlockType blockType;
    private int decoderId;
    private boolean lastDecoder;

    public DecoderBlock() {} // Empty constructor for deserialization

    public DecoderBlock(DecoderBlockType blockType, int decoderId, boolean lastDecoder)
    {
        this.blockType = blockType;
        this.decoderId = decoderId;
        this.lastDecoder = lastDecoder;
    }

    // Getters
    public DecoderBlockType getBlockType() {return blockType;}
    public int getDecoderId() {return decoderId;}
    public boolean getLastDecoder() {return lastDecoder;}

    @Override
    public String toString()
    {
        return blockType + "/" + decoderId;
    }
}
