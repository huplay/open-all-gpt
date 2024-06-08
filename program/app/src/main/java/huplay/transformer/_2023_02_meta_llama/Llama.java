package huplay.transformer._2023_02_meta_llama;

import huplay.config.Parameter;
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
    - RMS normalization
    - No biases, only weights
    - Query, key and value matrices are stored separately
    - 16 bit parameters (FLOAT16)

  @author Hunor Szegi
 */
public class Llama extends BaseTransformer
{
    Parameter tokenEmbeddings, normWeight;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "embed_tokens.weight", embeddingCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "norm.weight",         hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        return matrix(tokenEmbeddings).row(tokenId);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(normWeight), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(tokenEmbeddings)).getValues();

        return selectBestToken(logits, topK);
    }
}
