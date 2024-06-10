package transformer._2022_05_big_science_bloom;

import config.Parameter;
import math.dataType.matrix.Matrix;
import transformer.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
 * BLOOM decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class BloomAttentionLayer extends BaseAttentionLayer
{
    private float[] positionSlope;

    protected final List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    protected final List<List<Vector>> storedValues = new ArrayList<>(headCount);

    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

    public void loadParameters()
    {
        // Load parameters
        normWeight          = loadVector(NORM_WEIGHT,     "input_layernorm.weight",                hiddenSize);
        normBias            = loadVector(NORM_BIAS,       "input_layernorm.bias",                  hiddenSize);
        queryKeyValueWeight = loadMatrix(VERTICAL_WEIGHT, "self_attention.query_key_value.weight", hiddenSize * 3, hiddenSize);
        queryKeyValueBias   = loadVector(BIAS,            "self_attention.query_key_value.bias",   hiddenSize * 3);
        projectionWeight    = loadMatrix(VERTICAL_WEIGHT, "self_attention.dense.weight",           hiddenSize, hiddenSize);
        projectionBias      = loadVector(BIAS,            "self_attention.dense.bias",             hiddenSize);

        for (int i = 0; i < headCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);

        // Calculate the slope for the position embedding
        positionSlope = new float[headCount];
        float step = 1f / headCount;
        for (int i = 0; i < headCount; i++)
        {
            positionSlope[i] = step * (i + 1);
        }
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);
        }

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = hiddenState.multiplyByTransposed(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryKeyValuesByHead = queryKeyValue.split(headCount);

        // Matrix for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            Vector queryKeyValueByHead = queryKeyValuesByHead.row(head);

            // Split the query/key/value
            Matrix split = queryKeyValueByHead.split(3);
            Vector queryByHead = split.row(0);
            Vector keyByHead = split.row(1);
            Vector valueByHead = split.row(2);

            storedKeys.get(head).add(keyByHead);
            storedValues.get(head).add(valueByHead);

            // Store the keys and values (these will be available while the following tokens will be processed)
            int storedSize = storedKeys.get(head).size();

            // Calculate the scores
            Vector scores = Vector.emptyVector(hiddenState.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(head).get(pos);
                float score = queryByHead.dotProduct(relatedKey);

                // Position embedding at score
                score = score - positionSlope[head] * (storedSize - pos - 1);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(head).get(pos);
                Vector multipliedValue = relatedValue.multiply(scores.get(pos));

                Vector actualValue = valueAggregate.row(head);
                valueAggregate.setRow(head, actualValue.add(multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }
}
