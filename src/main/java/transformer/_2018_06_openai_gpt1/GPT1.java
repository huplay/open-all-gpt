package transformer._2018_06_openai_gpt1;

import config.Parameter;
import transformer.BaseTransformer;
import math.dataType.vector.Vector;

import static config.ParameterType.*;

/**
  OpenAI GPT-1 transformer

  The first GPT was released in June 2018.
  Publication: https://paperswithcode.com/paper/improving-language-understanding-by

  Difference to the original transformer:
    - Learned position embedding (not sinusoid)
    - GELU activation function (instead of ReLU)

  @author Hunor Szegi
 */
public class GPT1 extends BaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings;

    public void loadParameters()
    {
        tokenEmbeddings    = loadMatrix(EMBEDDING, "tokens_embed.weight",    tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING, "positions_embed.weight", contextSize, hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Find the position embedding of the position
        Vector positionEmbedding = matrix(positionEmbeddings).row(pos);

        // Return the addition of the hidden state and the position embedding
        return hiddenState.add(positionEmbedding);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
