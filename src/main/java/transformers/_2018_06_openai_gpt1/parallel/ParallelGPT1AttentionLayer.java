package transformers._2018_06_openai_gpt1.parallel;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.parallel.ParallelBaseAttentionLayer;

import java.util.ArrayList;
import java.util.List;

import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * OpenAI GPT-1 decoder (attention block) (Parallel implementation)
 * @author Hunor Szegi
 */
public class ParallelGPT1AttentionLayer extends ParallelBaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, projectionWeight, projectionBias;

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
    }

    public Matrix processParallel(Matrix inputHiddenState)
    {
        // Attention
        Matrix hiddenState = attentionParallel(inputHiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        // Normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        return hiddenState;
    }

    public Vector process(Vector inputHiddenState)
    {
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

        // Normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        return hiddenState;
    }

    private Matrix attentionParallel(Matrix hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Matrix queryKeyValue = hiddenState.multiply(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.addBroadcast(vector(queryKeyValueBias));

        // Split the query/key/value
        Matrix query = queryKeyValue.part(3, 0);
        Matrix key = queryKeyValue.part(3, 1);
        Matrix value = queryKeyValue.part(3, 2);

        // Collector of the attention results for all heads
        List<Matrix> valueAggregate = new ArrayList<>(headCount);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Split the query, key and value vectors into pieces for all heads
            Matrix queryByHead = query.part(headCount, head);
            Matrix keyByHead = key.part(headCount, head);
            Matrix valueByHead = value.part(headCount, head);

            // Store the keys and values (these will be available while the following tokens will be processed)
            store(head, keyByHead, valueByHead);

            // Process the core of the attention mechanism (scaled dot product attention)
            Matrix attentionResult = scaledDotProductAttentionParallel(
                                            queryByHead,
                                            getStoredKeys(head),
                                            getStoredValues(head));

            // Add the result to the collector for the actual head
            valueAggregate.add(attentionResult);
        }

        // Join the results of all heads
        hiddenState = MATH.joinMatrices(valueAggregate);

        // Projection neural layer
        hiddenState = hiddenState.multiply(matrix(projectionWeight));
        hiddenState = hiddenState.addBroadcast(vector(projectionBias));

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = hiddenState.multiply(matrix(queryKeyValueWeight));
        queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

        // Split the query/key/value
        Vector queries = queryKeyValue.part(3, 0);
        Vector keys = queryKeyValue.part(3, 1);
        Vector values = queryKeyValue.part(3, 2);

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Split the query, key and value vectors into pieces for all heads
            Vector query = queries.part(headCount, head);
            Vector key = keys.part(headCount, head);
            Vector value = values.part(headCount, head);

            // Store the keys and values (these will be available while the following tokens will be processed)0
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

    private Matrix scaledDotProductAttentionParallel(Matrix queries, Matrix keys, Matrix values)
    {
        Matrix attention = queries.multiplyByTransposed(keys);
        attention = attention.multiply(attentionScale);
        MATH.applyCausalMask(attention);
        attention = MATH.softmax(attention);
        attention = attention.multiply(values);

        return attention;
    }

    private Vector scaledDotProductAttention(Vector query, Matrix keys, Matrix values)
    {
        int tokenCount = keys.getRowCount();

        // Score all tokens using the actual query and the keys, multiplying by the scale
        Vector scores = emptyVector(tokenCount);
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedKey = keys.row(pos);

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
            Vector relatedValue = values.row(pos);
            float score = scores.get(pos);

            // Multiply the values by the score and sum up
            Vector scoredValue = relatedValue.multiply(score);
            result = result.add(scoredValue);
        }

        return result;
    }
}
