package huplay.transformer._2021_06_eleuther_gptj;

import huplay.config.Parameter;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import static huplay.config.ParameterType.*;
import static huplay.MathUtilProvider.MATH;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJNeuralNetLayer extends BaseNeuralNetLayer
{
    Parameter normWeight, normBias, layer1Weight, layer1Bias, layer2Weight, layer2Bias;

    public void loadParameters()
    {
        normWeight   = loadVector(NORM_WEIGHT,     "ln_f.weight",       hiddenSize);
        normBias     = loadVector(NORM_BIAS,       "ln_f.bias",         hiddenSize);
        layer1Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.fc_in.weight",  intermediateSize, hiddenSize);
        layer1Bias   = loadVector(BIAS,            "mlp.fc_in.bias",    intermediateSize);
        layer2Weight = loadMatrix(VERTICAL_WEIGHT, "mlp.fc_out.weight", hiddenSize, intermediateSize);
        layer2Bias   = loadVector(BIAS,            "mlp.fc_out.bias",   hiddenSize);
    }

    public Vector process(Vector inputHiddenState, Vector residualState)
    {
        // Normalisation
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = MATH.addVectors(residualState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <intermediateSize> neurons (usually 4 * <hiddenSize>) (using gelu activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer1Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer1Bias));

        for (int i = 0; i < intermediateSize; i++)
        {
            float activation =  MATH.gelu(hiddenState.get(i));
            hiddenState.set(i, activation);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(layer2Weight));
        hiddenState = MATH.addVectors(hiddenState, vector(layer2Bias));

        return hiddenState;
    }
}
