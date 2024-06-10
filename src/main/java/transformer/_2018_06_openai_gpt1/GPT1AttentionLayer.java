package transformer._2018_06_openai_gpt1;

import config.Parameter;
import math.dataType.matrix.Matrix;
import transformer.BaseAttentionLayer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
 * OpenAI GPT-1 decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class GPT1AttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

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
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);

            // Normalization
            hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);
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
            Vector actualQuery = queryByHead.row(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

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
