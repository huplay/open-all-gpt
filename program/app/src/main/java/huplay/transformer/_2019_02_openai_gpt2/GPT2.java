package huplay.transformer._2019_02_openai_gpt2;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;

/**
  OpenAI GPT-2 transformer

  Differences to GPT-1:
    - The normalization is used at the beginning of the attention and mlp blocks
    - Final normalization is added after the last decoder

  (The normalization before the first decoder's attention block gives more numerical stability at larger models.)

 * @author Hunor Szegi
 */
public class GPT2 extends BaseTransformer
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

        // Add the position embedding to hidden state
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
