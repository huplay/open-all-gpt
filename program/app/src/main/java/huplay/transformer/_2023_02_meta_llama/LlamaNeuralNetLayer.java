package huplay.transformer._2023_02_meta_llama;

import huplay.transformer.TransformerUtil;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * Meta Llama decoder implementation
 *
 * @author Hunor Szegi
 */
public class LlamaNeuralNetLayer extends BaseNeuralNetLayer
{
    public void loadParameters()
    {
        loadVector(MLP_NORM_WEIGHT, "post_attention_layernorm.weight", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.gate_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.up_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_3_WEIGHT, "mlp.down_proj.weight", hiddenSize, feedForwardSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = RMSLayerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Feed parallel two layers with the same input
        Vector hiddenState1 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        Vector hiddenState2 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, TransformerUtil.swiglu(hiddenState1.get(neuron)));
        }

        // Multiply the two outputs
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, hiddenState1.get(neuron) * hiddenState2.get(neuron));
        }

        // Use the third layer (no activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState1, matrix(MLP_3_WEIGHT));

        return hiddenState;
    }
}
