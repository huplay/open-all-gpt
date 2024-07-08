package transformers._2023_02_meta_llama;

import config.Parameter;
import transformer.serial.BaseNeuralNetLayer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
 * Meta (Facebook) Llama decoder (neural net block) implementation
 *
 * @author Hunor Szegi
 */
public class LlamaNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, gateProjectionWeight, upProjectionWeight, downProjectionWeight;

    public void loadParameters()
    {
        normWeight           = loadVector(NORM_WEIGHT,     "post_attention_layernorm.weight", hiddenSize);
        gateProjectionWeight = loadMatrix(VERTICAL_WEIGHT, "mlp.gate_proj.weight",            intermediateSize, hiddenSize);
        upProjectionWeight   = loadMatrix(VERTICAL_WEIGHT, "mlp.up_proj.weight",              intermediateSize, hiddenSize);
        downProjectionWeight = loadMatrix(VERTICAL_WEIGHT, "mlp.down_proj.weight",            hiddenSize, intermediateSize);
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
        // Feed parallel the gate and up layers with the same input
        Vector gateState = hiddenState.multiplyByTransposed(matrix(gateProjectionWeight));
        Vector upState = hiddenState.multiplyByTransposed(matrix(upProjectionWeight));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float activation = swiglu(gateState.get(neuron));
            gateState.set(neuron, activation);
        }

        // Fuse the two by multiplying the outputs
        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float fused = gateState.get(neuron) * upState.get(neuron);
            gateState.set(neuron, fused);
        }

        // Use the down layer (no activation function)
        hiddenState = gateState.multiplyByTransposed(matrix(downProjectionWeight));

        return hiddenState;
    }
}
