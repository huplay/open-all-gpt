package huplay.transformer._2018_01_google_transformer;

import huplay.transformer.TransformerUtil;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.util.Vector;

import static huplay.AppNetworkClient.UTIL;
import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * Decoder implementation of the original decoder-only Transformer architecture created by Google Brain
 *
 * @author Hunor Szegi
 */
public class GoogleTransformerNeuralNetLayer extends BaseNeuralNetLayer
{
    public void loadParameters()
    {
        loadVector(MLP_NORM_WEIGHT, "ln_2.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "ln_2.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.c_fc.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_1_BIAS, "mlp.c_fc.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.c_proj.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_2_BIAS, "mlp.c_proj.bias", hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        //  Normalisation
        hiddenState = layerNorm(hiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, TransformerUtil.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
