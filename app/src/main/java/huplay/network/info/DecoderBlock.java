package huplay.network.info;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class DecoderBlock
{
    private DecoderBlockType blockType;
    private int decoderId;

    public DecoderBlock() {} // Empty constructor for deserialization

    public DecoderBlock(DecoderBlockType blockType, int decoderId)
    {
        this.blockType = blockType;
        this.decoderId = decoderId;
    }

    // Getters
    public DecoderBlockType getBlockType() {return blockType;}
    public int getDecoderId() {return decoderId;}

    @Override
    public String toString()
    {
        return "DecoderBlock{" + blockType + "/" + decoderId + "}";
    }
}
