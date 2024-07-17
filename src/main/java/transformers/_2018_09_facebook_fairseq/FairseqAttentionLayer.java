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
        Vector queries, keys, values;

        if (splitQueryKeyValue)
        {
            // Calculate the query-key-value vectors for the actual token
            Vector queryKeyValue = hiddenState.multiplyByTransposed(matrix(queryKeyValueWeight));
            queryKeyValue = queryKeyValue.add(vector(queryKeyValueBias));

            // Slice the query/key/value
            queries = queryKeyValue.part(3, 0);
            keys = queryKeyValue.part(3, 1);
            values = queryKeyValue.part(3, 2);
        }
        else
        {
            // Calculate the query-key-value vectors for the actual token
            queries = hiddenState.multiplyByTransposed(matrix(queryWeight));
            queries = queries.add(vector(queryBias));

            keys = hiddenState.multiplyByTransposed(matrix(keyWeight));
            keys = keys.add(vector(keyBias));

            values = hiddenState.multiplyByTransposed(matrix(valueWeight));
            values = values.add(vector(valueBias));
        }

        // Apply the attention scale
        queries = queries.multiply(attentionScale);

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
