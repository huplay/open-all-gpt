package huplay.transformer._2022_05_big_science_bloom;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;

/**
 * BLOOM decoder implementation
 *
 * @author Hunor Szegi
 */
public class BloomNeuralNetLayer extends BaseNeuralNetLayer
{
    // Declare the used parameters (id, parameter type):
    Parameter NORM_WEIGHT = par("post_attention_layernorm.weight", NORMALIZATION_WEIGHT);
    Parameter NORM_BIAS = par("post_attention_layernorm.bias", NORMALIZATION_BIAS);
    Parameter NEURAL_LAYER_1_WEIGHT = par("mlp.dense_h_to_4h.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_1_BIAS = par("mlp.dense_h_to_4h.bias", BIAS);
    Parameter NEURAL_LAYER_2_WEIGHT = par("mlp.dense_4h_to_h.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_2_BIAS = par("mlp.dense_4h_to_h.bias", BIAS);

    float[] positionSlope;

    List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    List<List<Vector>> storedValues = new ArrayList<>(headCount);

    public void loadParameters()
    {
        loadVector(NORM_WEIGHT, hiddenSize);
        loadVector(NORM_BIAS, hiddenSize);
        loadMatrix(NEURAL_LAYER_1_WEIGHT, feedForwardSize, hiddenSize);
        loadVector(NEURAL_LAYER_1_BIAS, feedForwardSize);
        loadMatrix(NEURAL_LAYER_2_WEIGHT, hiddenSize, feedForwardSize);
        loadVector(NEURAL_LAYER_2_BIAS, hiddenSize);
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
