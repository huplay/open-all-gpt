package transformers._2018_09_facebook_fairseq;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.serial.BaseAttentionLayer;

import java.util.List;

import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * Meta (Facebook) XGLM decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class FairseqAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, normBias, queryKeyValueWeight, queryKeyValueBias, queryWeight, queryBias, keyWeight,
            keyBias, valueWeight, valueBias, projectionWeight, projectionBias;

    boolean splitQueryKeyValue;

    public void loadParameters()
    {
        splitQueryKeyValue = config.getBooleanValue("splitQueryKeyValue", false);

        if (splitQueryKeyValue)
        {
            queryKeyValueWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.in_proj.weight", hiddenSize, hiddenSize + 3);
            queryKeyValueBias   = loadVector(BIAS,            "self_attn.in_proj.bias",   hiddenSize + 3);
        }
        else
        {
            queryWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
            queryBias   = loadVector(BIAS,            "self_attn.q_proj.bias",   hiddenSize);
            keyWeight   = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight", hiddenSize, hiddenSize);
            keyBias     = loadVector(BIAS,            "self_attn.k_proj.bias",   hiddenSize);
            valueWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight", hiddenSize, hiddenSize);
            valueBias   = loadVector(BIAS,            "self_attn.v_proj.bias",   hiddenSize);
        }

        normWeight       = loadVector(NORM_WEIGHT,     "self_attn_layer_norm.weight", hiddenSize);
        normBias         = loadVector(NORM_BIAS,       "self_attn_layer_norm.bias",   hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.out_proj.weight",   hiddenSize, hiddenSize);
        projectionBias   = loadVector(BIAS,            "self_attn.out_proj.bias",     hiddenSize);

        // Calculate the attention scale
        attentionScale = 1 / sqrt(headSize);
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
        Vector query, key, value;

        if (splitQueryKeyValue)
        {
            // Calculate the query-key-value vectors for the actual token
            Vector queryKeyValue = hiddenState.multiplyByTransposed(matrix(queryKeyValueWeight));
            queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

            // Split the query/key/value
            Matrix split = queryKeyValue.split(3);
            query = split.row(0);
            key = split.row(1);
            value = split.row(2);
        }
        else
        {
            // Calculate the query-key-value vectors for the actual token
            query = hiddenState.multiplyByTransposed(matrix(queryWeight));
            query = query.add(vector(queryBias));

            key = hiddenState.multiplyByTransposed(matrix(keyWeight));
            key = key.add(vector(keyBias));

            value = hiddenState.multiplyByTransposed(matrix(valueWeight));
            value = value.add(vector(valueBias));
        }

        // Apply the attention scale
        query = query.multiply(attentionScale);

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Store the keys and values (these will be available while the following tokens will be processed)
            store(head, keyByHead, valueByHead);

            // Process the core of the attention mechanism (scaled dot product attention)
            Vector attentionResult = dotProductAttention(
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
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }

    private Vector dotProductAttention(Vector query, List<Vector> keys, List<Vector> values)
    {
        int tokenCount = keys.size();

        // Score all tokens using the actual query and the keys, multiplying by the scale
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
