package network.info.output;

import math.dataType.DataType;

public class HiddenStateOutput implements Output
{
    private DataType floatType;
    private float[] hiddenState;

    public HiddenStateOutput() {} // Empty constructor for deserialization

    public HiddenStateOutput(DataType floatType, float[] hiddenState)
    {
        this.floatType = floatType;
        this.hiddenState = hiddenState;
    }

    // Getters
    public DataType getFloatType() {return floatType;}
    public float[] getHiddenState() {return hiddenState;}

    @Override
    public String toString()
    {
        return "HiddenStateOutput{" + floatType.name() + ", " + hiddenState.length + "}";
    }
}
