package huplay.transformer._2019_02_openai_gpt2;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;
import static huplay.math.BasicMathUtility.sqrt;

/**
 * OpenAI GPT-2 decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPT2AttentionLayer extends BaseAttentionLayer
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
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = MATH.addVectors(inputHiddenState, hiddenState);
        }

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = MATH.mulVectorByMatrix(hiddenState, matrix(queryKeyValueWeight));
        queryKeyValue = MATH.addVectors(queryKeyValue, vector(queryKeyValueBias));

        // Split the query/key/value
        Matrix split = MATH.splitVector(queryKeyValue, 3);
        Vector query = split.row(0);
        Vector key = split.row(1);
        Vector value = split.row(2);

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByHead = MATH.splitVector(key, headCount);
        Matrix valueByHead = MATH.splitVector(value, headCount);

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
                float score = MATH.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).row(head);
                Vector multipliedValue = MATH.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setRow(head, MATH.addVectors(valueAggregate.row(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = MATH.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(projectionWeight));
        hiddenState = MATH.addVectors(hiddenState, vector(projectionBias));

        return hiddenState;
    }
}
