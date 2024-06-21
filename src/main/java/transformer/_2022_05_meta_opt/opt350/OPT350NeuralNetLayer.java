package transformer._2022_05_meta_opt.opt350;

import math.dataType.vector.Vector;
import transformer._2022_05_meta_opt.OPTNeuralNetLayer;

import static math.MathUtil.MATH;

/**
 * Meta (Facebook) OPT-350 decoder (neural net block) implementation
 * Only the process method is re-implemented, the rest is inherited from the standard OPT implementation
 *
 * @author Hunor Szegi
 */
public class OPT350NeuralNetLayer extends OPTNeuralNetLayer
{
    public Vector process(Vector inputHiddenState)
    {
        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        // Normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        return hiddenState;
    }
}
