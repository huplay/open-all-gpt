package transformer._2021_06_eleutherai_gptj;

import config.Parameter;
import math.dataType.matrix.Matrix;
import transformer.BaseNeuralNetLayer;
import math.dataType.vector.Vector;

import static config.ParameterType.*;
import static config.ParameterType.BIAS;

/**
 * EleutherAI GPT-J decoder (neural net block) implementation
 *
 * @author Hunor Szegi
 */
public class GPTJNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.fc_in.weight",  intermediateSize, hiddenSize);
        layer1Bias   = loadVector(BIAS,            "mlp.fc_in.bias",    intermediateSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.fc_out.weight", hiddenSize, intermediateSize);
        layer2Bias   = loadVector(BIAS,            "mlp.fc_out.bias",   hiddenSize);
    }

    public Vector process(Vector hiddenStateCompound)
    {
        // Split the input hidden states
        Matrix input = hiddenStateCompound.split(3);
        Vector inputHiddenState = input.row(0);
        Vector hiddenState = input.row(1);
        Vector attentionOutputHiddenState = input.row(2);

        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = hiddenState.multiplyByTransposed(matrix(layer1Weight));
        hiddenState = hiddenState.add(vector(layer1Bias));

        for (int i = 0; i < intermediateSize; i++)
        {
            float activation = gelu(hiddenState.get(i));
            hiddenState.set(i, activation);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = hiddenState.multiplyByTransposed(matrix(layer2Weight));
        hiddenState = hiddenState.add(vector(layer2Bias));

        // Add the three input states
        hiddenState = hiddenState.add(inputHiddenState);
        hiddenState = hiddenState.add(attentionOutputHiddenState);

        return hiddenState;
    }
}
