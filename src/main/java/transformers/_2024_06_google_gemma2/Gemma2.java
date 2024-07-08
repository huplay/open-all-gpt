package transformers._2024_06_google_gemma2;

import config.Parameter;
import math.dataType.BrainFloat16;
import math.dataType.DataType;
import math.dataType.vector.Vector;
import transformer.serial.BaseTransformer;

import static config.ParameterType.EMBEDDING;
import static config.ParameterType.NORM_WEIGHT;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
  Google Gemma transformer

  Google released two Gemma models on 21 Feb 2024. Publication: https://arxiv.org/abs/2403.08295

  The architecture is almost identical to the Llama models.

  Differences to Llama:
    - Llama 2 introduced the GQA attention, where the same key/value is used at multiple queries (Grouped Query).
      Gemma 2B is an edge case of the GQA, where the same key/value is used at all queries. (MQA: Multi Query)
    - Extra input normalization: division by sqrt of the hidden size
    - The RSM normalization adds 1 to the weight (bias)
    - GELU activation function (same as at GPT-2 and most of the models)
    - The rotary position embedding is applied in sliced arrangement (not interleaved) (same as at EleutherAI GPT-NeoX)
    - TODO: Be careful: https://github.com/huggingface/transformers/pull/29402

  Gemma-2 was released on 27 June 2024: https://blog.google/technology/developers/google-gemma-2
  Differences to the original Gemma:
    - Distilled training (but it has no effect on inference)
    - GQA attention (Same as at Llama)
    - Sparse attention (Every second decoder looks back only to the previous 4096 tokens, can't reach the max 8096)
    - Logit soft-capping (Attention logits are capped at 50.0, final logits are at 30.0)
    - Additional normalization around
  // TODO: Add Gemma 2

  @author Hunor Szegi
 */
public class Gemma2 extends BaseTransformer
{
    Parameter tokenEmbeddings, normWeight;

    float normConstant;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "embed_tokens.weight", tokenCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "norm.weight",         hiddenSize);

        // Constant for extra input normalization (downcast to BrainFloat16 and back to Float32)
        normConstant = BrainFloat16.of(sqrt(hiddenSize)).getFloatValue();
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        hiddenState = Vector.of(DataType.FLOAT_32, hiddenState.getValues());

        // Extra input normalization: division by sqrt of the hidden size
        return hiddenState.multiply(normConstant);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(normWeight), epsilon, 1f);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
