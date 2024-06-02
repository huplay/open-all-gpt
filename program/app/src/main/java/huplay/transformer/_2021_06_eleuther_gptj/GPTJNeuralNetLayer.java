package huplay.transformer._2021_06_eleuther_gptj;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;
import static huplay.MathUtilProvider.MATH;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJNeuralNetLayer extends BaseNeuralNetLayer
{
    // Declare the used parameters (id, parameter type):
    Parameter NORM_WEIGHT = par("ln_f.weight", NORMALIZATION_WEIGHT);
    Parameter NORM_BIAS = par("ln_f.bias", NORMALIZATION_BIAS);
    Parameter NEURAL_LAYER_1_WEIGHT = par("mlp.fc_in.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_1_BIAS = par("mlp.fc_in.bias", BIAS);
    Parameter NEURAL_LAYER_2_WEIGHT = par("mlp.fc_out.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_2_BIAS = par("mlp.fc_out.bias", BIAS);

    int maxAttentionSize;

    public void loadParameters()
    {
        loadVector(NORM_WEIGHT, hiddenSize);
        loadVector(NORM_BIAS, hiddenSize);
        loadMatrix(NEURAL_LAYER_1_WEIGHT, hiddenSize, feedForwardSize);
        loadVector(NEURAL_LAYER_1_BIAS, feedForwardSize);
        loadMatrix(NEURAL_LAYER_2_WEIGHT, feedForwardSize, hiddenSize);
        loadVector(NEURAL_LAYER_2_BIAS, hiddenSize);
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
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(NEURAL_LAYER_1_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(NEURAL_LAYER_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(NEURAL_LAYER_2_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(NEURAL_LAYER_2_BIAS));

        return hiddenState;
    }
}
