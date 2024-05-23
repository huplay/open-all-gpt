package huplay.transformer._2021_06_eleuther_gptj;

import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJAttentionLayer extends BaseAttentionLayer
{
    private int maxAttentionSize;

    public void loadParameters()
    {
        // attn.bias BOOL: 1x1x2048x2048
        loadVector(ATT_NORM_WEIGHT, "ln_1.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "ln_1.bias", hiddenSize);
        loadMatrix(ATT_QUERY_WEIGHT, "attn.q_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_KEY_WEIGHT, "attn.k_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_VALUE_WEIGHT, "attn.v_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_PROJ_WEIGHT, "attn.out_proj.weight", hiddenSize, hiddenSize);

        maxAttentionSize = 256; // TODO: Move sparse attention to logic, not as config
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalisation
        Vector hiddenState = layerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));
        Vector key = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        Vector value = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = UTIL.splitVector(query, headCount);
        Matrix keyByHead = UTIL.splitVector(key, headCount);
        Matrix valueByHead = UTIL.splitVector(value, headCount);

        // Position embedding (RoPE)
        applyPosition(query, key);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Used only at sparse attention:
        /*if (storedSize > maxAttentionSize)
        {
            // Topping the maximum attention size we can drop the oldest stored values
            storedKeys.remove(0);
            storedValues.remove(0);
        }*/

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
                scores.set(pos, UTIL.dotProduct(actualQuery, relatedKey));
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
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        //hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }

    protected void applyPosition(Vector query, Vector key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query.get(i);
            query.set(i, query0 * x - query.get(i + 1) * y);
            query.set(i + 1, query0 * y - query.get(i + 1) * x);

            // Rotate key
            float key0 = key.get(i);
            key.set(i, key0 * x - key.get(i + 1) * y);
            key.set(i + 1, key0 * y - key.get(i + 1) * x);
        }
    }
}
