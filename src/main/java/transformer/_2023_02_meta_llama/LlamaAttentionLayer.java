package transformer._2023_02_meta_llama;

import config.Parameter;
import math.dataType.matrix.Matrix;
import transformer.BaseAttentionLayer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.NORM_WEIGHT;
import static config.ParameterType.VERTICAL_WEIGHT;
import static math.BasicMathUtility.*;

/**
 * Meta Llama decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class LlamaAttentionLayer extends BaseAttentionLayer
{
    Parameter queryWeight, keyWeight, valueWeight, normWeight, projectionWeight;

    int kvHeadCount;
    int kvHeadSize;

    public void loadParameters()
    {
        // Read the optional config for the "key/value head" count
        // (At the original attention (MHA) there was only a single kind of head. Call it "query head" from now on.)
        // If the "query head" is different to the "key/value head" count, we are using Grouped Query Attention (GQA)
        kvHeadCount = config.getIntOptional("num_key_value_heads", headCount);
        kvHeadSize = headCount / kvHeadCount;

        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight", hiddenSize / kvHeadSize, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight", hiddenSize / kvHeadSize, hiddenSize);
        normWeight       = loadVector(NORM_WEIGHT,     "input_layernorm.weight",  hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);

        // Calculate the attention dividend
        this.attentionDividend = sqrt(headSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon);

        if (kvHeadSize == 1)
        {
            // Multi Head Attention (MHA)
            hiddenState = attention(hiddenState);
        }
        else
        {
            // Grouped Query Attention (GQA)
            hiddenState = groupedQueryAttention(hiddenState);
        }

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);
        }

        return hiddenState;
    }

    protected Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));
        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByHead = key.split(headCount);
        Matrix valueByHead = value.split(headCount);

        // Position embedding (RoPE)
        applyPosition(query, key);

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
                float score = actualQuery.dotProduct(relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
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

    protected void applyPosition(Vector query, Vector key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query.get(i);
            query.set(i, query0 * x - query.get(i + 1) * y);
            query.set(i + 1, query0 * y - query.get(i + 1) * x);

            // Rotate key
            float key0 = key.get(i);
            key.set(i, key0 * x - key.get(i + 1) * y);
            key.set(i + 1, key0 * y - key.get(i + 1) * x);
        }
    }

    protected Vector groupedQueryAttention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));

        // The key and value matrices are smaller (less head count) than the query matrix
        Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);
        Matrix keyByGroup = key.split(headCount / kvHeadSize);
        Matrix valueByGroup = value.split(headCount / kvHeadSize);

        // Position embedding (RoPE)
        applyGroupedPosition(query, key);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByGroup);
        storedValues.add(valueByGroup);
        int storedSize = storedKeys.size();

        // Matrix for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            int group = head % kvHeadCount;

            // Calculate the scores
            Vector actualQuery = queryByHead.row(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).row(group);
                float score = actualQuery.dotProduct(relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).row(group);
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

    protected void applyGroupedPosition(Vector query, Vector key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query.get(i);
            query.set(i, query0 * x - query.get(i + 1) * y);
            query.set(i + 1, query0 * y - query.get(i + 1) * x);

            if (i < hiddenSize / kvHeadSize)
            {
                // Rotate key
                float key0 = key.get(i);
                key.set(i, key0 * x - key.get(i + 1) * y);
                key.set(i + 1, key0 * y - key.get(i + 1) * x);
            }
        }
    }
}
