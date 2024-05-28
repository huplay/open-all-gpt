package huplay.network.info.input;

import huplay.dataType.DataType;

public class HiddenStateInput implements Input
{
    private DataType floatType;
    private float[] hiddenState;

    public HiddenStateInput()
    {
    }

    public HiddenStateInput(DataType floatType, float[] hiddenState)
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
        return "HiddenStateInput{" + floatType.name() + ", " + hiddenState.length + "}";
    }
}
