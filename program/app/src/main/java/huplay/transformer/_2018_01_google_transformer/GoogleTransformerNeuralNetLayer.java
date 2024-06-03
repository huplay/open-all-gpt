package huplay.transformer._2018_01_google_transformer;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * Decoder implementation of the original decoder-only Transformer architecture created by Google Brain
 *
 * @author Hunor Szegi
 */
public class GoogleTransformerNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter NORM_WEIGHT, NORM_BIAS, LAYER_1_WEIGHT, LAYER_1_BIAS, LAYER_2_WEIGHT, LAYER_2_BIAS;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("ln_2.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("ln_2.bias", NORMALIZATION_BIAS, hiddenSize);
        LAYER_1_WEIGHT = loadMatrix("mlp.c_fc.weight", HORIZONTAL_WEIGHT, hiddenSize, feedForwardSize);
        LAYER_1_BIAS = loadVector("mlp.c_fc.bias", BIAS, feedForwardSize);
        LAYER_2_WEIGHT = loadMatrix("mlp.c_proj.weight", HORIZONTAL_WEIGHT, feedForwardSize, hiddenSize);
        LAYER_2_BIAS = loadVector("mlp.c_proj.bias", BIAS, hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        //  Normalisation
        hiddenState = MATH.layerNorm(hiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(LAYER_1_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(LAYER_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(LAYER_2_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(LAYER_2_BIAS));

        return hiddenState;
    }
}
