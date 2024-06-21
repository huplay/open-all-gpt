package transformer._2022_02_eleutherai_gptneox;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.BaseNeuralNetLayer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
 * EleutherAI GPT-NeoX decoder (neural net block) implementation
 *
 * @author Hunor Szegi
 */
public class GPTNeoXNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "post_attention_layernorm.weight", hiddenSize);
        normBias     = loadVector(NORM_BIAS,       "post_attention_layernorm.bias",   hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.dense_h_to_4h.weight",        hiddenSize, intermediateSize);
        layer1Bias   = loadVector(BIAS,            "mlp.dense_h_to_4h.bias",          intermediateSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.dense_4h_to_h.weight",        intermediateSize, hiddenSize);
        layer2Bias   = loadVector(BIAS,            "mlp.dense_4h_to_h.bias",          hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = hiddenState.multiply(matrix(layer1Weight));
        hiddenState = hiddenState.add(vector(layer1Bias));

        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float activation = geluFast(hiddenState.get(neuron));
            hiddenState.set(neuron, activation);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = hiddenState.multiply(matrix(layer2Weight));
        hiddenState = hiddenState.add(vector(layer2Bias));

        return hiddenState;
    }
}
