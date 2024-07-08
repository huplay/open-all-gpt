package transformers._2019_02_openai_gpt2.parallel;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.parallel.ParallelBaseNeuralNetLayer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
 * OpenAI GPT-2 decoder (neural net block) (Parallel implementation)
 *
 * @author Hunor Szegi
 */
public class GPT2NeuralNetLayer extends ParallelBaseNeuralNetLayer
{
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,       "ln_2.weight",       hiddenSize);
        normBias     = loadVector(NORM_BIAS,         "ln_2.bias",         hiddenSize);
        layer1Weight = loadMatrix(HORIZONTAL_WEIGHT, "mlp.c_fc.weight",   hiddenSize, intermediateSize);
        layer1Bias   = loadVector(BIAS,              "mlp.c_fc.bias",     intermediateSize);
        layer2Weight = loadMatrix(HORIZONTAL_WEIGHT, "mlp.c_proj.weight", intermediateSize, hiddenSize);
        layer2Bias   = loadVector(BIAS,              "mlp.c_proj.bias",   hiddenSize);
    }

    public Matrix processParallel(Matrix inputHiddenState)
    {
        // Normalization
        Matrix hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Neural layers
        hiddenState = neuralNetParallel(hiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        return hiddenState;
    }

    private Matrix neuralNetParallel(Matrix hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = hiddenState.multiply(matrix(layer1Weight));
        hiddenState = hiddenState.addBroadcast(vector(layer1Bias));

        for (var i = 0; i < hiddenState.getRowCount(); i++)
        {
            for (int neuron = 0; neuron < intermediateSize; neuron++)
            {
                float activation = gelu(hiddenState.getValue(i, neuron));
                hiddenState.setValue(i, neuron, activation);
            }
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = hiddenState.multiply(matrix(layer2Weight));
        hiddenState = hiddenState.addBroadcast(vector(layer2Bias));

        return hiddenState;
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = hiddenState.multiply(matrix(layer1Weight));
        hiddenState = hiddenState.add(vector(layer1Bias));

        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float activation = gelu(hiddenState.get(neuron));
            hiddenState.set(neuron, activation);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = hiddenState.multiply(matrix(layer2Weight));
        hiddenState = hiddenState.add(vector(layer2Bias));

        return hiddenState;
    }
}
