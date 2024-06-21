package transformer._2023_09_mistralai_mistral;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.BaseTransformer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
  Mistral transformer


  Differences to GPT-2:


  @author Hunor Szegi
 */
public class Mistral extends BaseTransformer
{
    Parameter tokenEmbeddings, lmHead, outputNormWeight;

    public void loadParameters()
    {
        tokenEmbeddings  = loadMatrix(EMBEDDING,   "model.embed_tokens.weight", tokenCount, hiddenSize);
        lmHead           = loadMatrix(NORM_WEIGHT, "lm_head.weight",            tokenCount, hiddenSize);
        outputNormWeight = loadVector(NORM_WEIGHT, "model.norm.weight",         hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Input normalization
        return MATH.RMSLayerNorm(hiddenState, vector(lmHead), epsilon);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(outputNormWeight), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
