package huplay.quantization.qlora;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class QloraQuantState
{
    @JsonAlias({"quant_type"})
    private String quantType;

    @JsonAlias({"blocksize", "block_size"})
    private int blockSize;

    @JsonAlias({"dtype"})
    private String dataType;

    @JsonAlias({"shape"})
    private int[] shape;

    // Getters
    public String getQuantType() {return quantType;}
    public int getBlockSize() {return blockSize;}
    public String getDataType() {return dataType;}
    public int[] getShape() {return shape;}
}
