package huplay.transformer._2023_02_meta_llama;

import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
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
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, "embed_tokens.weight", embeddingCount + 3, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "norm.weight", hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        return matrix(TOKEN_EMBEDDINGS).getVector(token);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), epsilon);

        return determineOutputToken(hiddenState, topK);
    }
}
