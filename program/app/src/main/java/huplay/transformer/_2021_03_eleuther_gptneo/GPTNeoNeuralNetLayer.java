package huplay.transformer._2021_03_eleuther_gptneo;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * EleutherAI GPT-NEO decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTNeoNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "ln_2.weight",       hiddenSize);
        normBias     = loadVector(NORM_BIAS,       "ln_2.bias",         hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.c_fc.weight",   intermediateSize, hiddenSize);
        layer1Bias   = loadVector(BIAS,            "mlp.c_fc.bias",     intermediateSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.c_proj.weight", hiddenSize, intermediateSize);
        layer2Bias   = loadVector(BIAS,            "mlp.c_proj.bias",   hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer1Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer1Bias));

        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer2Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer2Bias));

        return hiddenState;
    }
}
