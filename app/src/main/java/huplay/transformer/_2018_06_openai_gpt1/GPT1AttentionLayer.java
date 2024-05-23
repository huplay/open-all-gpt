package huplay.transformer._2018_06_openai_gpt1;

import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * OpenAI GPT-1 decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPT1AttentionLayer extends BaseAttentionLayer
{
    public void loadParameters()
    {
        loadVector(ATT_NORM_WEIGHT, "ln_1.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "ln_1.bias", hiddenSize);
        loadMatrix(ATT_QUERY_KEY_VALUE_WEIGHT, "attn.c_attn.weight", hiddenSize, hiddenSize * 3);
        loadVector(ATT_QUERY_KEY_VALUE_BIAS, "attn.c_attn.bias", hiddenSize * 3);
        loadMatrix(ATT_PROJ_WEIGHT, "attn.c_proj.weight", hiddenSize, hiddenSize);
        loadVector(ATT_PROJ_BIAS, "attn.c_proj.bias", hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        // Normalisation
        hiddenState = layerNorm(hiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = UTIL.addVectors(queryKeyValue, vector(ATT_QUERY_KEY_VALUE_BIAS));

        // Split the query/key/value
        Matrix split = UTIL.splitVector(queryKeyValue, 3);
        Vector query = split.getVector(0);
        Vector key = split.getVector(1);
        Vector value = split.getVector(2);

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = UTIL.splitVector(query, headCount);
        Matrix keyByHead = UTIL.splitVector(key, headCount);
        Matrix valueByHead = UTIL.splitVector(value, headCount);

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
            Vector actualQuery = queryByHead.getVector(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).getVector(head);
                float score = UTIL.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).getVector(head);
                Vector multipliedValue = UTIL.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setVector(head, UTIL.addVectors(valueAggregate.getVector(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = UTIL.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }
}
