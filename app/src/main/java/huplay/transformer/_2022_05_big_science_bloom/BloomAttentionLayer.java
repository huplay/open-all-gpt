package huplay.transformer._2022_05_big_science_bloom;

import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

import static huplay.dataType.matrix.Matrix.emptyMatrix;
import static huplay.transformer.TransformerUtil.*;
import static huplay.config.ParameterType.*;

/**
 * BLOOM decoder implementation
 *
 * @author Hunor Szegi
 */
public class BloomAttentionLayer extends BaseAttentionLayer
{
    private float[] positionSlope;

    protected final List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    protected final List<List<Vector>> storedValues = new ArrayList<>(headCount);

    public void loadParameters()
    {
        for (int i = 0; i < headCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }

        // Load parameters
        loadVector(ATT_NORM_WEIGHT, "input_layernorm.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "input_layernorm.bias", hiddenSize);
        loadMatrix(ATT_QUERY_KEY_VALUE_WEIGHT, "self_attention.query_key_value.weight", hiddenSize * 3, hiddenSize);
        loadVector(ATT_QUERY_KEY_VALUE_BIAS, "self_attention.query_key_value.bias", hiddenSize * 3);
        loadMatrix(ATT_PROJ_WEIGHT, "self_attention.dense.weight", hiddenSize, hiddenSize);
        loadVector(ATT_PROJ_BIAS, "self_attention.dense.bias", hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);

        // Calculate the slope for the position embedding
        positionSlope = new float[headCount];
        float step = 1f / headCount;
        for (int i = 0; i < headCount; i++)
        {
            positionSlope[i] = step * (i + 1);
        }
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalisation
        Vector hiddenState = layerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = UTIL.addVectors(queryKeyValue, vector(ATT_QUERY_KEY_VALUE_BIAS));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryKeyValuesByHead = UTIL.splitVector(queryKeyValue, headCount);

        // Declaration of the variable for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            Vector queryKeyValueByHead = queryKeyValuesByHead.getVector(head);

            // Split the query/key/value
            Matrix split = UTIL.splitVector(queryKeyValueByHead, 3);
            Vector queryByHead = split.getVector(0);
            Vector keyByHead = split.getVector(1);
            Vector valueByHead = split.getVector(2);

            storedKeys.get(head).add(keyByHead);
            storedValues.get(head).add(valueByHead);

            // Store the keys and values (these will be available while the following tokens will be processed)
            int storedSize = storedKeys.get(head).size();

            // Calculate the scores
            Vector scores = Vector.emptyVector(hiddenState.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(head).get(pos);
                float score = UTIL.dotProduct(queryByHead, relatedKey);

                // Position embedding at score
                score = score - positionSlope[head] * (storedSize - pos - 1);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(head).get(pos);
                Vector multipliedValue = UTIL.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate.setVector(head, UTIL.addVectors(valueAggregate.getVector(head), multipliedValue));
            }
        }

        // Concatenate the results for all heads
        hiddenState = UTIL.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }
}
