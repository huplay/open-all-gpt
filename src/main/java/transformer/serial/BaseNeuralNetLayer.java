package transformer.serial;

import config.Config;
import math.NeuralNetUtil;
import math.dataType.vector.Vector;

public abstract class BaseNeuralNetLayer extends BaseDecoderLayer implements NeuralNetUtil
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
