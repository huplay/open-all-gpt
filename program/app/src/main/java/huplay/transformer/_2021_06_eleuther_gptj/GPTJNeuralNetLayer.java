package huplay.transformer._2021_06_eleuther_gptj;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.config.ParameterType.*;
import static huplay.MathUtilProvider.MATH;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter NORM_WEIGHT, NORM_BIAS, LAYER_1_WEIGHT, LAYER_1_BIAS, LAYER_2_WEIGHT, LAYER_2_BIAS;

    int maxAttentionSize;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("ln_f.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("ln_f.bias", NORMALIZATION_BIAS, hiddenSize);
        LAYER_1_WEIGHT = loadMatrix("mlp.fc_in.weight", VERTICAL_WEIGHT, hiddenSize, feedForwardSize);
        LAYER_1_BIAS = loadVector("mlp.fc_in.bias", BIAS, feedForwardSize);
        LAYER_2_WEIGHT = loadMatrix("mlp.fc_out.weight", VERTICAL_WEIGHT, feedForwardSize, hiddenSize);
        LAYER_2_BIAS = loadVector("mlp.fc_out.bias", BIAS, hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        //float[] hiddenState = layerNorm(inputHiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);

        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

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
