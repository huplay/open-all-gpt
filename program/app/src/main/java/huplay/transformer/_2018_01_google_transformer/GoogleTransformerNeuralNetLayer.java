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
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,       "ln_2.weight",       hiddenSize);
        normBias     = loadVector(NORM_BIAS,         "ln_2.bias",         hiddenSize);
        layer1Weight = loadMatrix(HORIZONTAL_WEIGHT, "mlp.c_fc.weight",   hiddenSize, intermediateSize);
        layer1Bias   = loadVector(BIAS,              "mlp.c_fc.bias",     intermediateSize);
        layer2Weight = loadMatrix(HORIZONTAL_WEIGHT, "mlp.c_proj.weight", intermediateSize, hiddenSize);
        layer2Bias   = loadVector(BIAS,              "mlp.c_proj.bias",   hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        //  Normalisation
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (4 * <hiddenSize>) (using gelu activation function)
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(layer1Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer1Bias));

        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(layer2Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer2Bias));

        return hiddenState;
    }
}
