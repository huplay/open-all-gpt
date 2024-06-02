package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;

/**
 * Meta Llama decoder implementation
 *
 * @author Hunor Szegi
 */
public class LlamaNeuralNetLayer extends BaseNeuralNetLayer
{
    // Declare the used parameters (id, parameter type):
    Parameter NORM_WEIGHT = par("post_attention_layernorm.weight", NORMALIZATION_WEIGHT);
    Parameter NEURAL_LAYER_1_WEIGHT = par("mlp.gate_proj.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_2_WEIGHT = par("mlp.up_proj.weight", VERTICAL_WEIGHT);
    Parameter NEURAL_LAYER_3_WEIGHT = par("mlp.down_proj.weight", VERTICAL_WEIGHT);

    public void loadParameters()
    {
        loadVector(NORM_WEIGHT, hiddenSize);
        loadMatrix(NEURAL_LAYER_1_WEIGHT, feedForwardSize, hiddenSize);
        loadMatrix(NEURAL_LAYER_2_WEIGHT, feedForwardSize, hiddenSize);
        loadMatrix(NEURAL_LAYER_3_WEIGHT, hiddenSize, feedForwardSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(NORM_WEIGHT), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Feed parallel two layers with the same input
        Vector hiddenState1 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(NEURAL_LAYER_1_WEIGHT));
        Vector hiddenState2 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(NEURAL_LAYER_2_WEIGHT));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, MATH.swiglu(hiddenState1.get(neuron)));
        }

        // Multiply the two outputs
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, hiddenState1.get(neuron) * hiddenState2.get(neuron));
        }

        // Use the third layer (no activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState1, matrix(NEURAL_LAYER_3_WEIGHT));

        return hiddenState;
    }
}
