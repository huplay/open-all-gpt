package huplay.transformer._2021_06_eleuther_gptj;

import huplay.transformer.TransformerUtil;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.config.ParameterType.*;
import static huplay.transformer.TransformerUtil.UTIL;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJNeuralNetLayer extends BaseNeuralNetLayer
{
    private int maxAttentionSize;

    public void loadParameters()
    {
        loadVector(MLP_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "ln_f.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.fc_in.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_1_BIAS, "mlp.fc_in.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.fc_out.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_2_BIAS, "mlp.fc_out.bias", hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        //float[] hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        Vector hiddenState = neuralNet(inputHiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, TransformerUtil.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
