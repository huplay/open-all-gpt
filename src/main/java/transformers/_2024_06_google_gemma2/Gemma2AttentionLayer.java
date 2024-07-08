package transformers._2024_06_google_gemma2;

import app.IdentifiedException;
import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import position.rotary.RotaryPositionEmbedding;
import transformer.serial.BaseAttentionLayer;

import java.util.List;

import static config.ParameterType.NORM_WEIGHT;
import static config.ParameterType.VERTICAL_WEIGHT;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * Google Gemma decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class Gemma2AttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, queryWeight, keyWeight, valueWeight, projectionWeight;

    RotaryPositionEmbedding positionEmbedding;

    public void loadParameters()
    {
        normWeight       = loadVector(NORM_WEIGHT,     "input_layernorm.weight",  hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight", kvSize, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight", kvSize, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);

        // Calculate the attention scale
        this.attentionScale = 1 / sqrt(kvSize);

        // Initialize the position embedding
        positionEmbedding = new RotaryPositionEmbedding(config, hiddenSize / headCount);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon, 1f);

        // Here the MHA and MQA cases are implemented separately to demonstrate the difference
        // You can check the Llama implementation where only the more general GQA case is implemented,
        // which covers MHA and MQA as edge cases
        if (kvHeadCount == headCount)
        {
            // Standard Multi Head Attention (MHA)
            hiddenState = attentionMHA(hiddenState);
        }
        else if (kvHeadCount == 1)
        {
            // Multi Query Attention (MQA)
            hiddenState = attentionMQA(hiddenState);
        }
        else
        {
            // It would be a non-edge case GQA, but there's no such a Gemma model
            throw new IdentifiedException("Gemma GQA case isn't implemented");
        }

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);
        }

        return hiddenState;
    }

    private Vector attentionMHA(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));

        // The key and value matrices are smaller (less head count) than the query matrix
        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // This is the position of the actually processed token:
        int pos = storedSize();

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Position embedding on the query and key
            positionEmbedding.applySliced(queryByHead.row(head), pos);
            positionEmbedding.applySliced(keyByHead.row(head), pos);

            // Store the keys and values (these will be available while the following tokens will be processed)
            store(head, keyByHead, valueByHead);

            // Process the core of the attention mechanism (scaled dot product attention)
            Vector attentionResult = scaledDotProductAttention(
                                            queryByHead.row(head),
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

    protected Vector attentionMQA(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));

        // The key and value matrices are smaller (less head count) than the query matrix
        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Split the query vector into pieces for all heads
        Matrix queryByHead = query.split(headCount);

        // No splitting for the key and value vectors, because at MQA the same vector is used for all heads

        // Position embedding on the key
        int pos = storedSize();
        positionEmbedding.applySliced(key, pos);

        // Store the keys and values (these will be available while the following tokens will be processed)
        store(0, key, value);

        // Matrix for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Position embedding on the query (separately within a head)
            Vector actualQuery = queryByHead.row(head);
            positionEmbedding.applySliced(actualQuery, pos);

            Vector attentionResult = scaledDotProductAttention(
                                            actualQuery,
                                            getStoredKeys(0),
                                            getStoredValues(0));

            valueAggregate.setRow(head, attentionResult);
        }

        // Concatenate the results for all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));

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
