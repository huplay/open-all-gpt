package huplay.transformer._2018_06_openai_gpt1;

import huplay.transformer.BaseTransformer;
import huplay.util.Vector;

import static huplay.AppNetworkClient.UTIL;
import static huplay.config.ParameterType.*;

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
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS)[token];

        // Add the position embedding to hidden state
        return UTIL.addVectors(hiddenState, matrix(POSITION_EMBEDDINGS)[pos]);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        return determineOutputToken(hiddenState, topK);
    }
}
