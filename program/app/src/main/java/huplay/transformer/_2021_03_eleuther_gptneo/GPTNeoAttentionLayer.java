package huplay.transformer._2021_03_eleuther_gptneo;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
 * EleutherAI GPT-NEO decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTNeoAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryWeight, keyWeight, valueWeight, projectionWeight, projectionBias;

    int maxAttentionSize;

    public void loadParameters()
    {
        normWeight       = loadVector(NORM_WEIGHT,     "ln_1.weight",                    hiddenSize);
        normBias         = loadVector(NORM_BIAS,       "ln_1.bias",                      hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.attention.q_proj.weight",   hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "attn.attention.k_proj.weight",   hiddenSize, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.attention.v_proj.weight",   hiddenSize, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "attn.attention.out_proj.weight", hiddenSize, hiddenSize);
        projectionBias   = loadVector(BIAS,             "attn.attention.out_proj.bias",  hiddenSize);

        maxAttentionSize = 256; // TODO: Move sparse attention to logic, not as config
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalisation
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
        // Calculate the query, key and value vectors for the actual token
        Vector query = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(queryWeight));
        Vector key = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(keyWeight));
        Vector value = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByHead = MATH.splitVector(key, headCount);
        Matrix valueByHead = MATH.splitVector(value, headCount);

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
        hiddenState = MATH.addVectors(hiddenState, vector(projectionBias));

        return hiddenState;
    }
}
