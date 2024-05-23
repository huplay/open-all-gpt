package huplay.transformer._2022_05_big_science_bloom;

import huplay.transformer.TransformerUtil;
import huplay.transformer.BaseNeuralNetLayer;
import huplay.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * BLOOM decoder implementation
 *
 * @author Hunor Szegi
 */
public class BloomNeuralNetLayer extends BaseNeuralNetLayer
{
    private float[] positionSlope;

    protected final List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    protected final List<List<Vector>> storedValues = new ArrayList<>(headCount);

    public void loadParameters()
    {
        loadVector(MLP_NORM_WEIGHT, "post_attention_layernorm.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "post_attention_layernorm.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.dense_h_to_4h.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_1_BIAS, "mlp.dense_h_to_4h.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.dense_4h_to_h.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_2_BIAS, "mlp.dense_4h_to_h.bias", hiddenSize);
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        hiddenState = neuralNet(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector neuralNet(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, TransformerUtil.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
