package huplay.network.info.output;

public class HiddenStateOutput implements Output
{
    private float[] hiddenState;

    public HiddenStateOutput() {} // Empty constructor for deserialization

    public HiddenStateOutput(float[] hiddenState)
    {
        this.hiddenState = hiddenState;
    }

    // Getter
    public float[] getHiddenState() {return hiddenState;}

    @Override
    public String toString()
    {
        return "HiddenStateOutput{" + hiddenState.length + "}";
    }
}
