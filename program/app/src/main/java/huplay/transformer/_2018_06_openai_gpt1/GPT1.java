package huplay.transformer._2018_06_openai_gpt1;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.config.Parameter.par;
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
    // Declare the used parameters (id, parameter type):
    Parameter TOKEN_EMBEDDINGS = par("tokens_embed.weight", EMBEDDINGS);
    Parameter POSITION_EMBEDDINGS = par("positions_embed.weight", EMBEDDINGS);

    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, contextSize, hiddenSize);
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
