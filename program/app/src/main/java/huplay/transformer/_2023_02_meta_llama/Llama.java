package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
import static huplay.config.Parameter.par;
import static huplay.config.ParameterType.*;

/**
  Meta Llama transformer

  Differences to GPT-2:
    - Rotary Position Embedding (RoPE)
    - Optionally Grouped Query Attention (GQA) (Only at certain models)
    - Two separate MLP layers, results multiplied and processed by a third layer
    - SwiGLU activation function
    - RMS normalisation
    - No biases, only weights
    - Query, key and value matrices are stored separately
    - 16 bit parameters (FLOAT16)

 * @author Hunor Szegi
 */
public class Llama extends BaseTransformer
{
    // Declare the used parameters (id, parameter type):
    Parameter TOKEN_EMBEDDINGS = par("embed_tokens.weight", EMBEDDINGS);
    Parameter NORM_WEIGHT = par("norm.weight", NORMALIZATION_WEIGHT);

    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, embeddingCount, hiddenSize);
        loadVector(NORM_WEIGHT, hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        return matrix(TOKEN_EMBEDDINGS).getRow(token);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(NORM_WEIGHT), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }
}
