package transformers._2023_02_meta_llama;

import config.Parameter;
import transformer.serial.BaseTransformer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
  Meta (Facebook) Llama transformer

  Llama was announced and released on 23 Feb 2023 by Meta (Facebook).
  Publication: https://arxiv.org/abs/2302.13971

  Differences to GPT-2:
    - Rotary Position Embedding (RoPE)
    - Two separate MLP layers, results multiplied and processed by a third layer
    - SwiGLU activation function
    - RMS normalization
    - No biases, only weights
    - Query, key and value matrices are stored separately
    - The weights are stored in vertical matrices (easier to execute the vector-matrix multiplication)
    - 16 bit parameters (FLOAT16)

  Llama 2 was announced and released on 18 July 2023. Publication: https://arxiv.org/abs/2307.09288

  Changes in Llama 2:
    - Grouped Query Attention (GQA) (Only at certain models)
    - The context lenght was increased from 2048 to 4096

  Llama 3 was announced and released on 18 April 2024. (Publication is expected in few months.)

  Changes in Llama 3:
    - Both models uses Grouped Query Attention (GQA)
    - Brain Float 16 data type (not Float 16)

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
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
