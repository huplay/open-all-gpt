package transformers._2023_09_mistralai_mistral;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.serial.BaseTransformer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
  Mistral transformer

  Mistral 7B v0.1 was released on 27 Sep 2023 by MistralAI.
  Publication: https://arxiv.org/abs/2310.06825

 The architecture is almost identical to the Llama models, even the parameter names are the same.

  Differences to Llama:
    - Untied embedding / un-embedding matrices (Separate parameters for token embeddings and generating logits.)
    - Sparse attention (only at v.0.1)

  Later they created an instruct variant (fine-trained answering to instructions),
  and there are v.02 and v.03 versions as well. (At the newer versions the sparse attention was dropped.)

  TODO: Sparse attention to be configurable "sliding_window": null
  TODO: Rope theta configurable

 @author Hunor Szegi
 */
public class Mistral extends BaseTransformer
{
    Parameter tokenEmbeddings, embeddingWeight, outputNormWeight;

    public void loadParameters()
    {
        tokenEmbeddings  = loadMatrix(EMBEDDING,   "model.embed_tokens.weight", tokenCount, hiddenSize);
        embeddingWeight  = loadMatrix(EMBEDDING,   "lm_head.weight",            tokenCount, hiddenSize);
        outputNormWeight = loadVector(NORM_WEIGHT, "model.norm.weight",         hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        return matrix(tokenEmbeddings).row(tokenId);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(outputNormWeight), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(embeddingWeight));

        return selectBestToken(logits, topK);
    }
}
