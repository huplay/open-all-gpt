package huplay.transformer._2019_02_openai_gpt2;

import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.*;
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
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getVector(token);

        // Add the position embedding to hidden state
        return MATH.addVectors(hiddenState, matrix(POSITION_EMBEDDINGS).getVector(pos));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);

        return determineOutputToken(hiddenState, topK);
    }
}
