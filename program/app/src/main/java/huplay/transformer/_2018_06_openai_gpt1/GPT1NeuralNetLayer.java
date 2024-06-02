package huplay.transformer._2018_06_openai_gpt1;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;

/**
 * OpenAI GPT-1 decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPT1NeuralNetLayer extends BaseNeuralNetLayer
{
    // Declare the used parameters (id, parameter type):
    Parameter NORM_WEIGHT = par("ln_2.weight", NORMALIZATION_WEIGHT);
    Parameter NORM_BIAS = par("ln_2.bias", NORMALIZATION_BIAS);
    Parameter NEURAL_LAYER_1_WEIGHT = par("mlp.c_fc.weight", HORIZONTAL_WEIGHT);
    Parameter NEURAL_LAYER_1_BIAS = par("mlp.c_fc.bias", BIAS);
    Parameter NEURAL_LAYER_2_WEIGHT = par("mlp.c_proj.weight", HORIZONTAL_WEIGHT);
    Parameter NEURAL_LAYER_2_BIAS = par("mlp.c_proj.bias", BIAS);

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
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(NEURAL_LAYER_1_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(NEURAL_LAYER_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, MATH.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(NEURAL_LAYER_2_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(NEURAL_LAYER_2_BIAS));

        return hiddenState;
    }
}
