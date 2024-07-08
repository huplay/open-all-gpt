package transformers._2019_02_openai_gpt2.parallel;

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
 * OpenAI GPT-2 decoder (attention block) (Parallel implementation)
 *
 * @author Hunor Szegi
 */
public class GPT2AttentionLayer extends ParallelBaseAttentionLayer
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
        // Normalization
        Matrix hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        hiddenState = attentionParallel(hiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

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

        // Concatenate the results of all heads
        hiddenState = flatten(valueAggregate);

        // Projection neural layer
        hiddenState = hiddenState.multiply(matrix(projectionWeight));
        hiddenState = hiddenState.addBroadcast(vector(projectionBias));

        return hiddenState;
    }

    private Matrix flatten(List<Matrix> matrices)
    {
        int size = 0;
        for (Matrix matrix : matrices) size += matrix.getColCount();

        var first = matrices.getFirst();
        Matrix result = Matrix.emptyMatrix(first.getInternalFloatType(), first.getRowCount(), first.getColCount());

        for (var i = 0; i < first.getRowCount(); i++)
        {
            result.setRow(i, flattenRow(matrices, size, i));
        }

        return result;
    }

    private Vector flattenRow(List<Matrix> matrices, int size, int index)
    {
        var result = emptyVector(size);

        var pos = 0;
        for (var matrix : matrices)
        {
            for (var i = 0; i < matrix.getColCount(); i++)
            {
                result.set(pos + i, matrix.getValue(index, i));
            }

            pos += matrix.getColCount();
        }

        return result;
    }

    private Matrix scaledDotProductAttentionParallel(Matrix queries, Matrix keys, Matrix values)
    {
        Matrix att = queries.multiplyByTransposed(keys);
        att = att.multiply(attentionScale);
        applyCausalMask(att);
        att = softmax(att);
        return att.multiply(values);
    }

    private Matrix softmax(Matrix matrix)
    {
        var result = emptyMatrix(matrix.getRowCount(), matrix.getColCount());

        for (var i = 0; i < matrix.getRowCount(); i++)
        {
            result.setRow(i, MATH.softmax(matrix.getVectorArray()[i]));
        }

        return result;
    }

    private void applyCausalMask(Matrix matrix)
    {
        for (var row = 0; row < matrix.getRowCount(); row++)
        {
            for (var col = 0; col < matrix.getColCount(); col++)
            {
                if (col > row)
                {
                    matrix.setValue(row, col, Float.NEGATIVE_INFINITY);
                }
            }
        }
    }

    public Vector process(Vector inputHiddenState)
    {
        // Normalization
        Vector hiddenState = MATH.layerNorm(inputHiddenState, vector(normWeight), vector(normBias), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Residual connection
        hiddenState = hiddenState.add(inputHiddenState);

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
