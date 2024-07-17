package transformers._2018_01_google_transformer;

import config.Parameter;
import position.sinusoid.SinusoidPositionEmbedding;
import transformer.serial.BaseTransformer;
import math.dataType.vector.Vector;

import static config.ParameterType.*;

/**
  Google Brain, the original decoder-only Transformer
  @author Hunor Szegi
 */
public class GoogleTransformer extends BaseTransformer
{
    Parameter tokenEmbeddings;
    SinusoidPositionEmbedding position = new SinusoidPositionEmbedding();

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING, "tokens_embed.weight", tokenCount, hiddenSize);

        position.initInterleaved(config, hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Position embedding
        hiddenState = position.apply(hiddenState, pos);

        return hiddenState;
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
