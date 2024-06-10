package quantization.qlora;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class QloraQuantState
{
    @JsonAlias({"quant_type"})
    private String quantType;

    @JsonAlias({"blocksize"})
    private int blockSize;

    @JsonAlias({"dtype"})
    private String dataType;

    @JsonAlias({"shape"})
    private int[] shape;

    @JsonAlias({"nested_blocksize"})
    private Integer nestedBlockSize;

    @JsonAlias({"nested_dtype"})
    private String nestedDataType;

    @JsonAlias({"nested_offset"})
    private float nestedOffset;

    // Getters
    public String getQuantType() {return quantType;}
    public int getBlockSize() {return blockSize;}
    public String getDataType() {return dataType;}
    public int[] getShape() {return shape;}
    public Integer getNestedBlockSize() {return nestedBlockSize;}
    public String getNestedDataType() {return nestedDataType;}
    public float getNestedOffset() {return nestedOffset;}
}
