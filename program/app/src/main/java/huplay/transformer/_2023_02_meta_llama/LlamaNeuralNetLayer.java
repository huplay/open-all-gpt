package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * Meta Llama decoder (neural net block) implementation
 *
 * @author Hunor Szegi
 */
public class LlamaNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, layer1Weight, layer2Weight, layer3Weight;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "post_attention_layernorm.weight", hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.gate_proj.weight",            intermediateSize, hiddenSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.up_proj.weight",              intermediateSize, hiddenSize);
        layer3Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.down_proj.weight",            hiddenSize, intermediateSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalization
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = inputHiddenState.add(hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Feed parallel two layers with the same input
        Vector hiddenState1 = hiddenState.multiplyByTransposed(matrix(layer1Weight));
        Vector hiddenState2 = hiddenState.multiplyByTransposed(matrix(layer2Weight));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float activation = MATH.swiglu(hiddenState1.get(neuron));
            hiddenState1.set(neuron, activation);
        }

        // Fuse the two by multiplying the outputs
        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float fused = hiddenState1.get(neuron) * hiddenState2.get(neuron);
            hiddenState1.set(neuron, fused);
        }

        // Use the third layer (no activation function)
        hiddenState = hiddenState1.multiplyByTransposed(matrix(layer3Weight));

        return hiddenState;
    }
}
