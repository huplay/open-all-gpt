package transformers._2022_05_big_science_bloom;

import config.Parameter;
import math.dataType.matrix.Matrix;
import position.alibi.AlibiPositionEmbedding;
import transformer.serial.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.List;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
 * BLOOM decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class BloomAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

    AlibiPositionEmbedding position = new AlibiPositionEmbedding();

    public void loadParameters()
    {
        // Load parameters
        normWeight          = loadVector(NORM_WEIGHT,     "input_layernorm.weight",                hiddenSize);
        normBias            = loadVector(NORM_BIAS,       "input_layernorm.bias",                  hiddenSize);
        queryKeyValueWeight = loadMatrix(VERTICAL_WEIGHT, "self_attention.query_key_value.weight", hiddenSize * 3, hiddenSize);
        queryKeyValueBias   = loadVector(BIAS,            "self_attention.query_key_value.bias",   hiddenSize * 3);
        projectionWeight    = loadMatrix(VERTICAL_WEIGHT, "self_attention.dense.weight",           hiddenSize, hiddenSize);
        projectionBias      = loadVector(BIAS,            "self_attention.dense.bias",             hiddenSize);

        // Calculate the attention scale
        attentionScale = 1 / sqrt(headSize);

        // Initialize the position embedder
        position.init(headCount);
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
        Vector queryKeyValue = hiddenState.multiplyByTransposed(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryKeyValuesByHead = queryKeyValue.split(headCount);

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            Vector queryKeyValueByHead = queryKeyValuesByHead.row(head);

            // Split the query/key/value
            Matrix split = queryKeyValueByHead.split(3);
            Vector queryByHead = split.row(0);
            Vector keyByHead = split.row(1);
            Vector valueByHead = split.row(2);

            // Store the keys and values (these will be available while the following tokens will be processed)
            store(head, keyByHead, valueByHead);

            // Process the core of the attention mechanism (dot product attention)
            Vector attentionResult = dotProductAttention(
                                            head,
                                            queryByHead,
                                            getStoredKeys(head),
                                            getStoredValues(head));

            // Add the result to the collector for the actual head
            valueAggregate.setRow(head, attentionResult);
        }

        // Concatenate the results of all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }

    private Vector dotProductAttention(int head, Vector query, List<Vector> keys, List<Vector> values)
    {
        int tokenCount = keys.size();

        // Score all tokens using the actual query and the keys
        Vector scores = emptyVector(tokenCount);
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedKey = keys.get(pos);

            // The core of the dot product attention (the scaling is applied only after the position embedding)
            float score = query.dotProduct(relatedKey);

            // Position embedding at score
            score = position.apply(score, head, tokenCount - pos);

            scores.set(pos, score * attentionScale);
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
