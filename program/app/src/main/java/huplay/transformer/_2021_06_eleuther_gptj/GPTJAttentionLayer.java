package huplay.transformer._2021_06_eleuther_gptj;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;
import static huplay.math.BasicMathUtility.*;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryWeight, keyWeight, valueWeight, projectionWeight;

    public void loadParameters()
    {
        normWeight       = loadVector(NORM_WEIGHT,     "ln_1.weight",          hiddenSize);
        normBias         = loadVector(NORM_BIAS,       "ln_1.bias",            hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.q_proj.weight",   hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "attn.k_proj.weight",   hiddenSize, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.v_proj.weight",   hiddenSize, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "attn.out_proj.weight", hiddenSize, hiddenSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalisation
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(queryWeight));
        Vector key = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(keyWeight));
        Vector value = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByHead = MATH.splitVector(key, headCount);
        Matrix valueByHead = MATH.splitVector(value, headCount);

        // Position embedding (RoPE)
        applyRotaryPosition(query, key);

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
            Vector actualQuery = queryByHead.getRow(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).getRow(head);
                scores.set(pos, MATH.dotProduct(actualQuery, relatedKey));
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).getRow(head);
                Vector multipliedValue = MATH.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setRow(head, MATH.addVectors(valueAggregate.getRow(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = MATH.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(projectionWeight));

        return hiddenState;
    }

    protected void applyRotaryPosition(Vector query, Vector key)
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
