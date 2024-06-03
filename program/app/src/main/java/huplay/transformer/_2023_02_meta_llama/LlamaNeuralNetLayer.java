package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * Meta Llama decoder implementation
 *
 * @author Hunor Szegi
 */
public class LlamaNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter NORM_WEIGHT, LAYER_1_WEIGHT, LAYER_2_WEIGHT, LAYER_3_WEIGHT;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("post_attention_layernorm.weight", NORMALIZATION_WEIGHT, hiddenSize);
        LAYER_1_WEIGHT = loadMatrix("mlp.gate_proj.weight", VERTICAL_WEIGHT, feedForwardSize, hiddenSize);
        LAYER_2_WEIGHT = loadMatrix("mlp.up_proj.weight", VERTICAL_WEIGHT, feedForwardSize, hiddenSize);
        LAYER_3_WEIGHT = loadMatrix("mlp.down_proj.weight", VERTICAL_WEIGHT, hiddenSize, feedForwardSize);
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
        Vector hiddenState1 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(LAYER_1_WEIGHT));
        Vector hiddenState2 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(LAYER_2_WEIGHT));

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
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState1, matrix(LAYER_3_WEIGHT));

        return hiddenState;
    }
}
