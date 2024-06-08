package huplay.transformer._2018_06_openai_gpt1;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.config.ParameterType.*;

/**
  OpenAI GPT-1 transformer

  Difference to the original transformer:
    - Learned position embedding (not sinusoid)

 * @author Hunor Szegi
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

        // Add the position embedding to hidden state
        return hiddenState.add(matrix(positionEmbeddings).row(pos));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
