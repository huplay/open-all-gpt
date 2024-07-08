package transformers._2018_09_facebook_fairseq;

import app.IdentifiedException;
import config.Parameter;
import math.dataType.vector.Vector;
import transformer.serial.BaseNeuralNetLayer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
 * Meta (Facebook) XGLM decoder (neural net block) implementation
 *
 * @author Hunor Szegi
 */
public class FairseqNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    String activationFunction = "relu";

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "final_layer_norm.weight", hiddenSize);
        normBias     = loadVector(NORM_BIAS,       "final_layer_norm.bias",   hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "fc1.weight",              intermediateSize, hiddenSize);
        layer1Bias   = loadVector(BIAS,            "fc1.bias",                intermediateSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "fc2.weight",              hiddenSize, intermediateSize);
        layer2Bias   = loadVector(BIAS,            "fc2.bias",                hiddenSize);

        activationFunction = config.getValue("activation_function");
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
        // Layer 1: <intermediateSize> neurons (4 * <hiddenSize>) (using GELU activation function)
        hiddenState = hiddenState.multiplyByTransposed(matrix(layer1Weight));
        hiddenState = hiddenState.add(vector(layer1Bias));

        for (int neuron = 0; neuron < intermediateSize; neuron++)
        {
            float activation = activation(hiddenState.get(neuron));
            hiddenState.set(neuron, activation);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = hiddenState.multiplyByTransposed(matrix(layer2Weight));
        hiddenState = hiddenState.add(vector(layer2Bias));

        return hiddenState;
    }

    private float activation(float value)
    {
        if (activationFunction.equals("relu"))
        {
            return relu(value);
        }
        else if (activationFunction.equals("gelu"))
        {
            return gelu(value);
        }
        else
        {
            throw new IdentifiedException("Unsupported activation function: " + activationFunction);
        }
    }
}
