package transformers._2023_02_meta_llama;

import config.Parameter;
import math.dataType.matrix.Matrix;
import position.rotary.RotaryPositionEmbedding;
import transformer.serial.BaseAttentionLayer;
import math.dataType.vector.Vector;

import java.util.List;

import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
 * Meta (Facebook) Llama decoder (attention block) implementation
 *
 * @author Hunor Szegi
 */
public class LlamaAttentionLayer extends BaseAttentionLayer
{
    Parameter normWeight, queryWeight, keyWeight, valueWeight, projectionWeight;

    RotaryPositionEmbedding positionEmbedding;

    public void loadParameters()
    {
        // Load parameters
        normWeight       = loadVector(NORM_WEIGHT,     "input_layernorm.weight",  hiddenSize);
        queryWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
        keyWeight        = loadMatrix(VERTICAL_WEIGHT, "self_attn.k_proj.weight", hiddenSize / headPerKvHead, hiddenSize);
        valueWeight      = loadMatrix(VERTICAL_WEIGHT, "self_attn.v_proj.weight", hiddenSize / headPerKvHead, hiddenSize);
        projectionWeight = loadMatrix(VERTICAL_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);

        // Calculate the attention scale
        this.attentionScale = 1 / sqrt(kvSize);

        // Initialize the position embedder
        positionEmbedding = new RotaryPositionEmbedding(config, kvSize);
    }

    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Normalization
        Vector hiddenState = MATH.RMSLayerNorm(inputHiddenState, vector(normWeight), epsilon);

        // Grouped Query Attention (GQA)
        hiddenState = attentionGQA(hiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);
        }

        return hiddenState;
    }

    protected Vector attentionGQA(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector queries = hiddenState.multiplyByTransposed(matrix(queryWeight));
        Vector keys = hiddenState.multiplyByTransposed(matrix(keyWeight));
        Vector values = hiddenState.multiplyByTransposed(matrix(valueWeight));

        // Collector of the attention results for all heads
        Matrix valueAggregate = emptyMatrix(headCount, headSize);

        // This is the position of the actually processed token:
        int pos = storedSize();

        // Score the previous tokens (including the actual), separately for all heads
        int head = 0;
        for (int kvHead = 0; kvHead < kvHeadCount; kvHead++)
        {
            // Get the query vector for the actual head
            Vector query = queries.part(headCount, head);

            for (int i = 0; i < headPerKvHead; i++)
            {
                // Get the key and value vectors for the actual key-value head
                Vector key = keys.part(kvHeadCount, i);
                Vector value = values.part(kvHeadCount, i);

                // Position embedding on the query and key
                positionEmbedding.applyInterleaved(query, pos);
                positionEmbedding.applyInterleaved(key, pos);

                // Store the keys and values (these will be available while the following tokens will be processed)
                store(kvHead, key, value);

                Vector attentionResult = scaledDotProductAttention(
                                                query,
                                                getStoredKeys(kvHead),
                                                getStoredValues(kvHead));

                valueAggregate.setRow(head, attentionResult);

                head++;
            }
        }

        // Concatenate the results of all heads
        hiddenState = valueAggregate.flatten();

        // Projection neural layer
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectionWeight));

        return hiddenState;
    }

    private Vector scaledDotProductAttention(Vector query, List<Vector> keys, List<Vector> values)
    {
        int tokenCount = keys.size();

        // Score all tokens using the actual query and the keys, multiplying by the scale
        Vector scores = emptyVector(tokenCount);
        for (int pos = 0; pos < tokenCount; pos++)
        {
            Vector relatedKey = keys.get(pos);

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
            Vector relatedValue = values.get(pos);
            float score = scores.get(pos);

            // Multiply the values by the score and sum up
            Vector scoredValue = relatedValue.multiply(score);
            result = result.add(scoredValue);
        }

        return result;
    }
}
