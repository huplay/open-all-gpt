package huplay.transformer._2018_01_google_transformer;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseAttentionLayer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;
import static huplay.math.BasicMathUtility.sqrt;

/**
 * Decoder implementation of the original decoder-only Transformer architecture created by Google Brain
 *
 * @author Hunor Szegi
 */
public class GoogleTransformerAttentionLayer extends BaseAttentionLayer
{
    Parameter NORM_WEIGHT, NORM_BIAS, QUERY_KEY_VALUE_WEIGHT, QUERY_KEY_VALUE_BIAS, PROJECTION_WEIGHT, PROJECTION_BIAS;

    public void loadParameters()
    {
        NORM_WEIGHT = loadVector("ln_1.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("ln_1.bias", NORMALIZATION_BIAS, hiddenSize);
        QUERY_KEY_VALUE_WEIGHT = loadMatrix("attn.c_attn.weight", HORIZONTAL_WEIGHT, hiddenSize, hiddenSize * 3);
        QUERY_KEY_VALUE_BIAS = loadVector("attn.c_attn.bias", BIAS, hiddenSize * 3);
        PROJECTION_WEIGHT = loadMatrix("attn.c_proj.weight", HORIZONTAL_WEIGHT, hiddenSize, hiddenSize);
        PROJECTION_BIAS = loadVector("attn.c_proj.bias", BIAS, hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        if (isInputOnly && lastDecoder) // During input token processing at the last decoder...
            return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest
        
        // Residual connection
        hiddenState = MATH.addVectors(inputHiddenState, hiddenState);

        // Normalisation
        hiddenState = MATH.layerNorm(hiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);
    
        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = MATH.mulVectorByMatrix(hiddenState, matrix(QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = MATH.addVectors(queryKeyValue, vector(QUERY_KEY_VALUE_BIAS));

        // Split the query/key/value
        Matrix split = MATH.splitVector(queryKeyValue, 3);
        Vector query = split.getRow(0);
        Vector key = split.getRow(1);
        Vector value = split.getRow(2);

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = MATH.splitVector(query, headCount);
        Matrix keyByHead = MATH.splitVector(key, headCount);
        Matrix valueByHead = MATH.splitVector(value, headCount);

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
            Vector scores = Vector.emptyVector(storedSize);

            Vector actualQuery = queryByHead.getRow(head);

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
        hiddenState = MATH.mulVectorByMatrix(hiddenState, matrix(PROJECTION_WEIGHT));
        hiddenState = MATH.addVectors(hiddenState, vector(PROJECTION_BIAS));

        return hiddenState;
    }
}
