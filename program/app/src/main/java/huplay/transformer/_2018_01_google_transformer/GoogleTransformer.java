package huplay.transformer._2018_01_google_transformer;

import huplay.config.Parameter;
import huplay.dataType.matrix.Matrix;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
import static huplay.config.ParameterType.*;
import static huplay.math.BasicMathUtility.exp;

/**
  Google Brain, the original decoder-only Transformer

  The encoder-decoder architecture was described in the famous "Attention Is All You Need" paper:
  June 2017: https://arxiv.org/abs/1706.03762

  The decoder-only variant was described in "Generating Wikipedia by Summarizing Long Sequences"
  Jan 2018: https://arxiv.org/abs/1801.10198

  Features:
    - Sinusoid position embedding added to the input at the beginning
    - Normalization is used at the end of the attention and feed-forward blocks
    - Residual connections at the attention and feed-forward blocks
    - Multi-head attention
    - Score dividend in the attention, which is calculated as sqrt(headSize)
    - Single layer projection at the end of the attention blocks
    - Feed-forward block has two layers (layer1: 4 * hiddenSize neurons, layer2: hiddenSize neurons)
    - GELU activation function (used only at the first feed-forward layer)
    - 32 bit parameters
    - query/key/value matrices are stored in a single matrix

   The weights of the trained model wasn't published.
   I don't know about any model which uses exactly the same architecture, so

 * @author Hunor Szegi
 */
public class GoogleTransformer extends BaseTransformer
{
    Parameter TOKEN_EMBEDDINGS;
    private Matrix positionMatrix = calculatePositionMatrix();

    public void loadParameters()
    {
        TOKEN_EMBEDDINGS = loadMatrix("tokens_embed.weight", EMBEDDINGS, tokenCount, hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getRow(token);

        // Position embedding
        for (int i = 0; i < hiddenState.size(); i++)
        {
            hiddenState.set(i, hiddenState.get(i) * positionMatrix.getValue(pos, i));
        }

        return hiddenState;
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }

    private Matrix calculatePositionMatrix()
    {
        Matrix positionMatrix = emptyMatrix(contextSize, hiddenSize);

        Vector positions = emptyVector(contextSize);
        for (int i = 0; i < contextSize; i++)
        {
            positions.set(i, i);
        }

        Vector progression = emptyVector(contextSize / 2);
        for (int i = 0; i < contextSize / 2; i++)
        {
            progression.set(i, exp(-i * Math.log(10000) / contextSize));
        }

        for (int pos = 0; pos < contextSize; pos++)
        {
            for (int k = 0; k < hiddenSize / 2; k++)
            {
                int i = 2 * k;
                positionMatrix.setValue(pos, i, (float) Math.sin(positions.get(i) * progression.get(k)));
                positionMatrix.setValue(pos, i + 1, (float) Math.sin(positions.get(i + 1) * progression.get(k)));
            }
        }

        return positionMatrix;
    }
}
