package huplay.transformer._2022_05_big_science_bloom;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * BLOOM decoder implementation
 *
 * @author Hunor Szegi
 */
public class BloomNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter NORM_WEIGHT, NORM_BIAS, LAYER_1_WEIGHT, NN_LAYER_1_BIAS, LAYER_2_WEIGHT, LAYER_2_BIAS;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("post_attention_layernorm.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("post_attention_layernorm.bias", NORMALIZATION_BIAS, hiddenSize);
        LAYER_1_WEIGHT = loadMatrix("mlp.dense_h_to_4h.weight", VERTICAL_WEIGHT, feedForwardSize, hiddenSize);
        NN_LAYER_1_BIAS = loadVector("mlp.dense_h_to_4h.bias", BIAS, feedForwardSize);
        LAYER_2_WEIGHT = loadMatrix("mlp.dense_4h_to_h.weight", VERTICAL_WEIGHT, hiddenSize, feedForwardSize);
        LAYER_2_BIAS = loadVector("mlp.dense_4h_to_h.bias", BIAS, hiddenSize);
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
        hiddenState = MATH.addVectors(hiddenState, vector(NN_LAYER_1_BIAS));

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
