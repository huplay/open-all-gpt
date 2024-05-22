package huplay.network.info.input;

import huplay.dataType.FloatType;

public class HiddenStateInput implements Input
{
    private FloatType floatType;
    private float[] hiddenState;

    public HiddenStateInput()
    {
    }

    public HiddenStateInput(FloatType floatType, float[] hiddenState)
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
        return "HiddenStateInput{" + floatType.name() + ", " + hiddenState.length + "}";
    }
}
