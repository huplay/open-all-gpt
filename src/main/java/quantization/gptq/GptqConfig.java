package quantization.gptq;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class GptqConfig
{
    // Bit size. (2, 3, 4, 8 or 16)
    @JsonAlias({"bits"})
    private int bits;

    @JsonAlias({"group_size"})
    private int groupSize;

    // Decreasing activation order (act order)
    @JsonAlias({"desc_act"})
    private boolean descAct;

    @JsonAlias({"sym"})
    // Symmetric or asymmetric quantization. At symmetric all zeros are 0, oth
    private boolean sym;

    // Getters
    public int getBits() {return bits;}
    public int getGroupSize() {return groupSize;}
    public boolean getDescAct() {return descAct;}
    public boolean getSym() {return sym;}
}
