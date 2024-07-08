package transformer.parallel;

import config.Config;
import math.NeuralNetUtil;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.serial.BaseDecoderLayer;

public abstract class ParallelBaseNeuralNetLayer extends BaseDecoderLayer implements NeuralNetUtil
{
    protected int intermediateSize;

    public abstract void loadParameters();

    public abstract Matrix processParallel(Matrix inputHiddenState);

    public abstract Vector process(Vector inputHiddenState);

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);
        this.intermediateSize = config.getIntermediateSize();

        loadParameters();
    }
}
