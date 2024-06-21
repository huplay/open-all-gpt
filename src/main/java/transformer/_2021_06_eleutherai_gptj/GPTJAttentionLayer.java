package transformer._2021_06_eleutherai_gptj;

import config.Parameter;
import math.dataType.matrix.Matrix;
import position.rotary.RotaryPositionEmbedding;
import transformer.BaseAttentionLayer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
 * EleutherAI GPT-J decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class GPTJAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryWeight, keyWeight, valueWeight, projectionWeight;

    RotaryPositionEmbedding positionEmbedding;

    public void loadParameters()
    {
        normWeight       = loadVector(NORM_WEIGHT,     "ln_1.weight",          hiddenSize);
        normBias         = loadVector(NORM_BIAS,       "ln_1.bias",            hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.q_proj.weight",   hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "attn.k_proj.weight",   hiddenSize, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "attn.v_proj.weight",   hiddenSize, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "attn.out_proj.weight", hiddenSize, hiddenSize);

        // Initialize the position embedder
        positionEmbedding = new RotaryPositionEmbedding(config, hiddenSize / headCount);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        Vector attentionOutputHiddenState = attention(hiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Join the input hidden state, hidden state and output hidden state (to pass all to the neural net block)
            hiddenState = MATH.joinVectors(inputHiddenState, hiddenState);
            hiddenState = MATH.joinVectors(hiddenState, attentionOutputHiddenState);
        }

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));
        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

        // Position embedding (RoPE)
        positionEmbedding.applyInterleaved(query, storedKeys.size());
        positionEmbedding.applyInterleaved(key, storedKeys.size());

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
                scores.set(pos, actualQuery.dotProduct(relatedKey));
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
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));

        return hiddenState;
    }
}