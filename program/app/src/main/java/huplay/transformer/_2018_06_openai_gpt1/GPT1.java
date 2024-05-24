package huplay.transformer._2018_06_openai_gpt1;

import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.config.ParameterType.*;
import static huplay.MathUtilProvider.MATH;

/**
  OpenAI GPT-1 transformer

  Difference to the original transformer:
    - Learned position embedding (not sinusoid)

 * @author Hunor Szegi
 */
public class GPT1 extends BaseTransformer
{
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, "tokens_embed.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "positions_embed.weight", contextSize, hiddenSize);
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
        return determineOutputToken(hiddenState, topK);
    }
}
