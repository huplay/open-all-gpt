package transformer._2023_09_mistralai_mistral;

import config.Parameter;
import math.dataType.matrix.Matrix;
import position.alibi.AlibiPositionEmbedding;
import transformer.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
 * Mistral decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class MistralAttentionLayer extends BaseAttentionLayer
{
    protected final List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    protected final List<List<Vector>> storedValues = new ArrayList<>(headCount);

    Parameter normWeight, queryWeight, keyWeight, valueWeight, projectionWeight;

    int kvHeadCount;
    int headPerKvHead;

    AlibiPositionEmbedding position = new AlibiPositionEmbedding();

    public void loadParameters()
    {
        // Read the optional config for the "key/value head" count
        // (At the original attention (MHA) there was only a single kind of head. Call it "query head" from now on.)
        // If the "query head" is different to the "key/value head" count, we are using Grouped Query Attention (GQA)
        kvHeadCount = config.getIntOptional("num_key_value_heads", headCount);
        headPerKvHead = headCount / kvHeadCount;

        // Load parameters
        normWeight       = loadVector(NORM_WEIGHT,     "input_layernorm.weight",  hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight", hiddenSize / headPerKvHead, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight", hiddenSize / headPerKvHead, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);

        for (int i = 0; i < headCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }

        // Initialize the position embedder
        position.init(headCount);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon);

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

    protected Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = hiddenState.multiplyByTransposed(matrix(queryWeight));

        // Split the query, key and value vectors into pieces for all heads
        Matrix queryByHead = query.split(headCount);

        // Store the keys and values (these will be available while the following tokens will be processed)

        int storedSize = storedKeys.size();

        // Matrix for collecting the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // Score the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // The key and value matrices are smaller (less head count) than the query matrix
            Vector key = hiddenState.multiplyByTransposed(matrix(keyWeight));
            Vector value = hiddenState.multiplyByTransposed(matrix(valueWeight));

            storedKeys.get(head).add(key);
            storedValues.get(head).add(value);

            int group = head % kvHeadCount;

            // Calculate the scores
            Vector actualQuery = queryByHead.row(head);

            Vector scores = Vector.emptyVector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(head).get(pos);
                float score = actualQuery.dotProduct(relatedKey);

                // Position embedding at score
                score = position.apply(score, head, storedSize - pos);

                // Divide the score by the attention dividend
                scores.set(pos, score);
            }

            // Scale the scores to values between 0 and 1
            scores = MATH.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(head).get(pos);
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
}
