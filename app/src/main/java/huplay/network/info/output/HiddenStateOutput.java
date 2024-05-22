package huplay.network.info.output;

import huplay.dataType.FloatType;

public class HiddenStateOutput implements Output
{
    private FloatType floatType;
    private float[] hiddenState;

    public HiddenStateOutput() {} // Empty constructor for deserialization

    public HiddenStateOutput(FloatType floatType, float[] hiddenState)
    {
        this.floatType = floatType;
        this.hiddenState = hiddenState;
    }

    // Getters
    public FloatType getFloatType() {return floatType;}
    public float[] getHiddenState() {return hiddenState;}

    @Override
    public String toString()
    {
        return "HiddenStateOutput{" + floatType.name() + ", " + hiddenState.length + "}";
    }
}
