package transformer._2020_05_openai_gpt3;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.BaseAttentionLayer;

import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * OpenAI GPT-3 decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class GPT3AttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

    boolean isLocalAttention;
    int maxLocalAttentionSize = 256;

    public void loadParameters()
    {
        normWeight          = loadVector(NORM_WEIGHT,       "ln_1.weight",        hiddenSize);
        normBias            = loadVector(NORM_BIAS,         "ln_1.bias",          hiddenSize);
        queryKeyValueWeight = loadMatrix(HORIZONTAL_WEIGHT, "attn.c_attn.weight", hiddenSize, hiddenSize * 3);
        queryKeyValueBias   = loadVector(BIAS,              "attn.c_attn.bias",   hiddenSize * 3);
        projectionWeight    = loadMatrix(HORIZONTAL_WEIGHT, "attn.c_proj.weight", hiddenSize, hiddenSize);
        projectionBias      = loadVector(BIAS,              "attn.c_proj.bias",   hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);

        // Every second decoder has local attention (if the decoderId is an odd number),
        // which means the attention size is capped. (It is called "sparse attention".)
        isLocalAttention = (decoderId % 2 != 0);
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
        Vector queryKeyValue = hiddenState.multiply(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

        // Split the query/key/value
        Matrix split = queryKeyValue.split(3);
        Vector query = split.row(0);
        Vector key = split.row(1);
        Vector value = split.row(2);

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

        // At local attention we can forget the stored keys/values for the too distant tokens (above limit)
        if (isLocalAttention && storedKeys.size() > maxLocalAttentionSize)
        {
            storedKeys.removeFirst();
            storedValues.removeFirst();
        }

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Matrix for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Calculate the scores
            Vector scores = emptyVector(storedSize);

            Vector actualQuery = queryByHead.row(head);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).row(head);
                float score = actualQuery.dotProduct(relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).row(head);
                Vector multipliedValue = relatedValue.multiply(scores.get(pos));

                Vector actualValue = valueAggregate.row(head);
                valueAggregate.setRow(head, actualValue.add(multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiply(matrix(projectionWeight));
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }
}
