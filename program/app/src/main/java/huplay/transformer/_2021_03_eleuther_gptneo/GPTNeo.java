package huplay.transformer._2021_03_eleuther_gptneo;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.ParameterType.*;

/**
  EleutherAI GPT-NEO transformer

  Differences to GPT-2:
    - Sparse decoders: Every second decoder uses local attention, using only the previous 256 tokens
    - No biases for the attention query/key/value matrices
    - query/key/value matrices are stored separately
    - No attention dividend, so the score isn't divided by a fixed value
    - The weights are stored in transposed matrices (easier to execute the vector-matrix multiplication)

  @author Hunor Szegi
 */
public class GPTNeo extends BaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings    = loadMatrix(EMBEDDING,   "wte.weight",  tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING,   "wpe.weight",  contextSize, hiddenSize);
        normWeight         = loadVector(NORM_WEIGHT, "ln_f.weight", hiddenSize);
        normBias           = loadVector(NORM_BIAS,   "ln_f.bias",   hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Position embedding
        return MATH.addVectors(hiddenState, matrix(positionEmbeddings).row(pos));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(tokenEmbeddings)).getValues();

        return selectBestToken(logits, topK);
    }
}
