package transformers._2020_05_openai_gpt3;

import config.Parameter;
import math.dataType.matrix.Matrix;
import transformer.serial.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.List;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
 * OpenAI GPT-3 decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class GPT3AttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

    boolean isLocalAttention;
    int maxLocalAttentionSize = 256;

    public void loadParameters()
    {
        normWeight          = loadVector(NORM_WEIGHT,       "ln_1.weight",        hiddenSize);
        normBias            = loadVector(NORM_BIAS,         "ln_1.bias",          hiddenSize);
        queryKeyValueWeight = loadMatrix(HORIZONTAL_WEIGHT, "attn.c_attn.weight", hiddenSize, hiddenSize * 3);
        queryKeyValueBias   = loadVector(BIAS,              "attn.c_attn.bias",   hiddenSize * 3);
        projectionWeight    = loadMatrix(HORIZONTAL_WEIGHT, "attn.c_proj.weight", hiddenSize, hiddenSize);
        projectionBias      = loadVector(BIAS,              "attn.c_proj.bias",   hiddenSize);

        // Calculate the attention scale
        attentionScale = 1 / sqrt(headSize);

        // Every second decoder has local attention (if the decoderId is an odd number),
        // which means the attention size is capped. (It is called "sparse attention".)
        isLocalAttention = (decoderId % 2 != 0);
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
        Vector queryKeyValue = hiddenState.multiply(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

        // Slice the query/key/value
        Vector queries = queryKeyValue.part(3, 0);
        Vector keys = queryKeyValue.part(3, 1);
        Vector values = queryKeyValue.part(3, 2);

        // At local attention we can forget the stored keys/values for the too distant tokens (above limit)
        if (isLocalAttention && storedSize() > maxLocalAttentionSize)
        {
            removeFirstStoredKey();
            removeFirstStoredValue();
        }

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

            // Process the core of the attention mechanism (scaled dot product attention)
            Vector attentionResult = scaledDotProductAttention(
                                            query,
                                            getStoredKeys(head),
                                            getStoredValues(head));

            // Add the result to the collector for the actual head
            valueAggregate.setRow(head, attentionResult);
        }

        // Concatenate the results of all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiply(matrix(projectionWeight));
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }

    private Vector scaledDotProductAttention(Vector query, List<Vector> keys, List<Vector> values)
    {
        int tokenCount = keys.size();

        // Score all tokens using the actual query and the keys, multiplying by the scale
        Vector scores = emptyVector(tokenCount);
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedKey = keys.get(pos);

            // The core of the scaled dot product attention
            float score = query.dotProduct(relatedKey) * attentionScale;
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
