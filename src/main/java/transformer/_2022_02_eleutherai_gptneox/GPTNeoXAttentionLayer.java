package transformer._2022_02_eleutherai_gptneox;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import position.rotary.RotaryPositionEmbedding;
import transformer.BaseAttentionLayer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
 * EleutherAI GPT-NeoX decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class GPTNeoXAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

    RotaryPositionEmbedding positionEmbedding;

    public void loadParameters()
    {
        normWeight          = loadVector(NORM_WEIGHT,     "input_layernorm.weight",           hiddenSize);
        normBias            = loadVector(NORM_BIAS,       "input_layernorm.bias",             hiddenSize);
        queryKeyValueWeight = loadMatrix(VERTICAL_WEIGHT, "attention.query_key_value.weight", hiddenSize, hiddenSize * 3);
        queryKeyValueBias   = loadVector(BIAS,            "attention.query_key_value.bias",   hiddenSize * 3);
        projectionWeight    = loadMatrix(VERTICAL_WEIGHT, "attention.dense.weight",           hiddenSize, hiddenSize);
        projectionBias      = loadVector(BIAS,            "attention.dense.bias",             hiddenSize);

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

        // Position embedding (RoPE)
        positionEmbedding.applySliced(query, storedKeys.size());
        positionEmbedding.applySliced(key, storedKeys.size());

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
