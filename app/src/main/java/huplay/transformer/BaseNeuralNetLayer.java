package huplay.transformer;

import huplay.config.Config;
import huplay.dataType.vector.Vector;

public abstract class BaseNeuralNetLayer extends BaseDecoderLayer
{
    protected int feedForwardSize;

    public abstract void loadParameters();

    public abstract Vector process(Vector inputHiddenState);

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

        return process(inputHiddenState);
    }

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);
        this.feedForwardSize = config.getFeedForwardSize();

        loadParameters();
    }
}
