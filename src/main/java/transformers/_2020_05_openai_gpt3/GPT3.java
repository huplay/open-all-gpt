package transformers._2020_05_openai_gpt3;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.serial.BaseTransformer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
  OpenAI GPT-3
  @author Hunor Szegi
 */
public class GPT3 extends BaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING, "wte.weight", tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING, "wpe.weight", contextSize, hiddenSize);
        normWeight = loadVector(NORM_WEIGHT, "ln_f.weight", hiddenSize);
        normBias = loadVector(NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token (this is the initial hidden state)
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Find the position embedding of the position
        Vector positionEmbedding = matrix(positionEmbeddings).row(pos);

        // Return the addition of the hidden state and the position embedding
        return hiddenState.add(positionEmbedding);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}