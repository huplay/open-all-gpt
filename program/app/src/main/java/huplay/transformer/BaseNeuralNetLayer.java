package huplay.transformer;

import huplay.config.Config;
import huplay.dataType.vector.Vector;

public abstract class BaseNeuralNetLayer extends BaseDecoderLayer
{
    protected int intermediateSize;

    public abstract void loadParameters();

    public abstract Vector process(Vector inputHiddenState);

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);
        this.intermediateSize = config.getIntermediateSize();

        loadParameters();
    }
}
