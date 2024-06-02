package huplay.transformer._2021_03_eleuther_gptneo;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
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
    // Declare the used parameters (id, parameter type):
    Parameter TOKEN_EMBEDDINGS = par("wte.weight", EMBEDDINGS);
    Parameter POSITION_EMBEDDINGS = par("wpe.weight", EMBEDDINGS);
    Parameter NORM_WEIGHT = par("ln_f.weight", NORMALIZATION_WEIGHT);
    Parameter NORM_BIAS = par("ln_f.bias", NORMALIZATION_BIAS);

    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, contextSize, hiddenSize);
        loadVector(NORM_WEIGHT, hiddenSize);
        loadVector(NORM_BIAS, hiddenSize);
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
        hiddenState = MATH.layerNorm(hiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }
}
