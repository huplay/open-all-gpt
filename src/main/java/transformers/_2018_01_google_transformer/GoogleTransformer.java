package transformers._2018_01_google_transformer;

import config.Parameter;
import position.sinusoid.SinusoidPositionEmbedding;
import transformer.serial.BaseTransformer;
import math.dataType.vector.Vector;

import static config.ParameterType.*;

/**
  Google Brain, the original decoder-only Transformer

  The encoder-decoder architecture was described in the famous "Attention Is All You Need" paper:
  June 2017: https://arxiv.org/abs/1706.03762

  The decoder-only variant was described in "Generating Wikipedia by Summarizing Long Sequences"
  Jan 2018: https://arxiv.org/abs/1801.10198

  Features:
    - Sinusoid position embedding added to the input at the beginning
      (But they created a variant of the original encoder-decoder architecture using learned position embedding as well.)
    - Normalization is used at the end of the attention and feed-forward blocks
    - Residual connections at the attention and feed-forward blocks
    - Multi-head attention
    - Scale the attention score by 1 / sqrt(headSize)
    - Single layer projection at the end of the attention blocks
    - Feed-forward block has two layers (layer1: 4 * hiddenSize neurons, layer2: hiddenSize neurons)
    - ReLU activation function (used only at the first feed-forward layer)
    - 32 bit parameters
    - query/key/value matrices are stored in a single matrix

  The weights of the trained model weren't published.
  I don't know about any model which uses exactly the same architecture,
  so currently you can't try this implementation.

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
