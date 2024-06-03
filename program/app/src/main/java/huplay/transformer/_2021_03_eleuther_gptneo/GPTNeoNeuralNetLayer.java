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
    Parameter NORM_WEIGHT, NORM_BIAS, LAYER_1_WEIGHT, LAYER_1_BIAS, LAYER_2_WEIGHT, LAYER_2_BIAS;

    int maxAttentionSize;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("ln_2.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("ln_2.bias", NORMALIZATION_BIAS, hiddenSize);
        LAYER_1_WEIGHT = loadMatrix("mlp.c_fc.weight", VERTICAL_WEIGHT, feedForwardSize, hiddenSize);
        LAYER_1_BIAS = loadVector("mlp.c_fc.bias", BIAS, feedForwardSize);
        LAYER_2_WEIGHT = loadMatrix("mlp.c_proj.weight", VERTICAL_WEIGHT, hiddenSize, feedForwardSize);
        LAYER_2_BIAS = loadVector("mlp.c_proj.bias", BIAS, hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(LAYER_1_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(LAYER_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(LAYER_2_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(LAYER_2_BIAS));

        return hiddenState;
    }
}
