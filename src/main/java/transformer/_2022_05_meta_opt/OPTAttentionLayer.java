package transformer._2022_05_meta_opt;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.BaseAttentionLayer;

import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * Meta (Facebook) OPT decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class OPTAttentionLayer extends BaseAttentionLayer
{
    public Parameter normWeight, normBias, queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias,
            projectionWeight, projectionBias;

    public void loadParameters()
    {
        normWeight       = loadVector(NORM_WEIGHT,     "self_attn_layer_norm.weight", hiddenSize);
        normBias         = loadVector(NORM_BIAS,       "self_attn_layer_norm.bias",   hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight",     hiddenSize, hiddenSize);
        queryBias        = loadVector(BIAS,            "self_attn.q_proj.bias",       hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight",     hiddenSize, hiddenSize);
        keyBias          = loadVector(BIAS,            "self_attn.k_proj.bias",       hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight",     hiddenSize, hiddenSize);
        valueBias        = loadVector(BIAS,            "self_attn.v_proj.bias",       hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.out_proj.weight",   hiddenSize, hiddenSize);
        projectionBias   = loadVector(BIAS,            "self_attn.out_proj.bias",     hiddenSize);

        // Calculate the attention dividend (will be used as multiplier, that's why it's 1/x)
        attentionDividend = 1 / sqrt(headSize);
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

    public Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));
        query = query.add(vector(queryBias));

        // Attention dividend applied on query
        query = query.multiply(attentionDividend);

        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        key = key.add(vector(keyBias));

        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));
        value = value.add(vector(valueBias));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

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
            Vector scores = emptyVector(storedSize);

            Vector actualQuery = queryByHead.row(head);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).row(head);
                float score = actualQuery.dotProduct(relatedKey);

                scores.set(pos, score);
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
        hiddenState = hiddenState.add(vector(projectionBias));

        return hiddenState;
    }
}
