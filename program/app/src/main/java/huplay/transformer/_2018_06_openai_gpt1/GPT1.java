package huplay.transformer._2018_06_openai_gpt1;

import huplay.config.Parameter;
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
    Parameter TOKEN_EMBEDDINGS, POSITION_EMBEDDINGS;

    public void loadParameters()
    {
        TOKEN_EMBEDDINGS = loadMatrix("tokens_embed.weight", EMBEDDINGS, tokenCount, hiddenSize);
        POSITION_EMBEDDINGS = loadMatrix("positions_embed.weight", EMBEDDINGS, contextSize, hiddenSize);
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
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }
}
