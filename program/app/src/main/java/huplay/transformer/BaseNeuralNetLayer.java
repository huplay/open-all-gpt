package huplay.transformer;

import huplay.config.Config;
import huplay.dataType.vector.Vector;

public abstract class BaseNeuralNetLayer extends BaseDecoderLayer
{
    protected int intermediateSize;

    public abstract void loadParameters();

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);
        this.intermediateSize = config.getIntermediateSize();

        loadParameters();
    }

    public Vector process(Vector inputHiddenState, Vector residualState)
    {
        // Empty implementation, but it can be overridden in subclasses
        // This one is called with the residual state (before the attention block),
        // but if we don't need that, we can leave it as it is, and override the other method (without residual state)
        return process(inputHiddenState);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Empty implementation, but it can be overridden in subclasses
        return inputHiddenState;
    }
}
