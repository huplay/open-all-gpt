package huplay.network.info.input;

public class HiddenStateInput implements Input
{
    private float[] hiddenState;

    public HiddenStateInput()
    {
    }

    public HiddenStateInput(float[] hiddenState)
    {
        this.hiddenState = hiddenState;
    }

    // Getter
    public float[] getHiddenState() {return hiddenState;}

    @Override
    public String toString()
    {
        return "HiddenStateInput{" + hiddenState.length + "}";
    }
}
