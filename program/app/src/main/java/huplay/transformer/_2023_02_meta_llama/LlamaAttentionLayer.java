package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.NORMALIZATION_WEIGHT;
import static huplay.config.ParameterType.VERTICAL_WEIGHT;
import static huplay.math.BasicMathUtility.*;

/**
 * Meta Llama decoder implementation
 *
 * @author Hunor Szegi
 */
public class LlamaAttentionLayer extends BaseAttentionLayer
{
    // Declare the used parameters (id, parameter type):
    Parameter QUERY_WEIGHT = par("self_attn.q_proj.weight", VERTICAL_WEIGHT);
    Parameter KEY_WEIGHT = par("self_attn.k_proj.weight", VERTICAL_WEIGHT);
    Parameter VALUE_WEIGHT = par("self_attn.v_proj.weight", VERTICAL_WEIGHT);
    Parameter NORM_WEIGHT = par("input_layernorm.weight", NORMALIZATION_WEIGHT);
    Parameter PROJECTION_WEIGHT = par("self_attn.o_proj.weight", VERTICAL_WEIGHT);

    int kvHeadCount;
    int kvHeadSize;

    public void loadParameters()
    {
        loadMatrix(QUERY_WEIGHT, hiddenSize, hiddenSize);

        // Read the optional config for the "key/value head" count
        // (At the original attention (MHA) there was only a single kind of head. Call it "query head" from now on.)
        // If the "query head" is different to the "key/value head" count, we are using Grouped Query Attention (GQA)
        kvHeadCount = config.getIntOptional("num_key_value_heads", headCount);
        kvHeadSize = headCount / kvHeadCount;
        loadMatrix(KEY_WEIGHT, hiddenSize, hiddenSize / kvHeadSize);
        loadMatrix(VALUE_WEIGHT, hiddenSize, hiddenSize / kvHeadSize);

        loadVector(NORM_WEIGHT, hiddenSize);
        loadMatrix(PROJECTION_WEIGHT, hiddenSize, hiddenSize);

        // Calculate the attention dividend
        this.attentionDividend = sqrt(headSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalisation
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(NORM_WEIGHT), epsilon);

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

        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    protected Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(QUERY_WEIGHT));
        Vector key = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(KEY_WEIGHT));
        Vector value = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByHead = MATH.splitVector(key, headCount);
        Matrix valueByHead = MATH.splitVector(value, headCount);

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
            Vector actualQuery = queryByHead.getRow(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).getRow(head);
                float score = MATH.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).getRow(head);
                Vector multipliedValue = MATH.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setRow(head, MATH.addVectors(valueAggregate.getRow(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = MATH.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(PROJECTION_WEIGHT));

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
        Vector query = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(QUERY_WEIGHT));

        // The key and value matrices are smaller (less head count) than the query matrix
        Vector key = MATH.mulVectorByMatrix(hiddenState, matrix(KEY_WEIGHT)); // TODO: Why it isn't transposed?
        Vector value = MATH.mulVectorByMatrix(hiddenState, matrix(VALUE_WEIGHT)); // TODO: Why it isn't transposed?

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByGroup = MATH.splitVector(key, headCount / kvHeadSize);
        Matrix valueByGroup = MATH.splitVector(value, headCount / kvHeadSize);

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
            Vector actualQuery = queryByHead.getRow(head);
            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos).getRow(group);
                float score = MATH.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos).getRow(group);
                Vector multipliedValue = MATH.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setRow(head, MATH.addVectors(valueAggregate.getRow(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = MATH.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(PROJECTION_WEIGHT));

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
