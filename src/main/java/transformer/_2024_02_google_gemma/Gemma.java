package transformer._2024_02_google_gemma;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.BaseTransformer;

import static math.MathUtil.MATH;
import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;

/**
  Google Gemma transformer

  Differences to Llama:
    - Extra input normalization: division by sqrt of the hidden size
    - GELU activation function (same as at GPT-2)

  @author Hunor Szegi
 */
public class Gemma extends BaseTransformer
{
    Parameter tokenEmbeddings, normWeight;

    float normConstant;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "embed_tokens.weight", tokenCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "norm.weight",         hiddenSize);

        normConstant = sqrt(hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

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
