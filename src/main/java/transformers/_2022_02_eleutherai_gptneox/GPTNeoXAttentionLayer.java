package transformers._2022_02_eleutherai_gptneox;

import config.Parameter;
import math.dataType.matrix.Matrix;
import position.rotary.RotaryPositionEmbedding;
import transformer.serial.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.List;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
 * EleutherAI GPT-NeoX decoder (attention block) implementation
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

        // Slice the query/key/value
        Vector queries = queryKeyValue.part(3, 0);
        Vector keys = queryKeyValue.part(3, 1);
        Vector values = queryKeyValue.part(3, 2);

        // Position embedding (RoPE)
        positionEmbedding.applySliced(queries, storedSize());
        positionEmbedding.applySliced(keys, storedSize());

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Get the part for the actual head of the query, key and value vectors
            Vector query = queries.part(headCount, head);
            Vector key = keys.part(headCount, head);
            Vector value = values.part(headCount, head);

            // Store the keys and values (these will be available while the following tokens will be processed)
            store(head, key, value);

            // Process the core of the attention mechanism (dot product attention)
            Vector attentionResult = dotProductAttention(
                                            query,
                                            getStoredKeys(head),
                                            getStoredValues(head));

            // Add the result to the collector for the actual head
            valueAggregate.setRow(head, attentionResult);
        }

        // Concatenate the results of all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));

        return hiddenState;
    }

    private Vector dotProductAttention(Vector query, List<Vector> keys, List<Vector> values)
    {
        int tokenCount = keys.size();

        // Score all tokens using the actual query and the keys
        Vector scores = emptyVector(tokenCount);
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedKey = keys.get(pos);

            // The core of the dot product attention
            float score = query.dotProduct(relatedKey);
            scores.set(pos, score);
        }

        // Normalize the scores into a range between 0 and 1
        scores = MATH.softmax(scores);

        // Apply the score on the values vectors
        Vector result = Vector.emptyVector(query.getFloatType(), query.size());
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedValue = values.get(pos);
            float score = scores.get(pos);

            // Multiply the values by the score and sum up
            Vector scoredValue = relatedValue.multiply(score);
            result = result.add(scoredValue);
        }

        return result;
    }
}
