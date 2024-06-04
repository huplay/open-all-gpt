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
    Parameter normWeight, layer1Weight, layer2Weight, layer3Weight;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "post_attention_layernorm.weight", hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.gate_proj.weight",            feedForwardSize, hiddenSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.up_proj.weight",              feedForwardSize, hiddenSize);
        layer3Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.down_proj.weight",            hiddenSize, feedForwardSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Feed parallel two layers with the same input
        Vector hiddenState1 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer1Weight));
        Vector hiddenState2 = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer2Weight));

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
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState1, matrix(layer3Weight));

        return hiddenState;
    }
}
