package huplay.transformer._2021_03_eleuther_gptneo;

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

 * @author Hunor Szegi
 */
public class GPTNeo extends BaseTransformer
{
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, "wte.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "wpe.weight", contextSize, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getRow(token);

        // Position embedding
        return MATH.addVectors(hiddenState, matrix(POSITION_EMBEDDINGS).getRow(pos));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);

        return determineOutputToken(hiddenState, topK);
    }
}
