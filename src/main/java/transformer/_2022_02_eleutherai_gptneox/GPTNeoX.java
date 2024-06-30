package transformer._2022_02_eleutherai_gptneox;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.BaseTransformer;

import static config.ParameterType.*;
import static config.ParameterType.NORM_BIAS;
import static math.MathUtil.MATH;

/**
  EleutherAI GPT-NeoX transformer

  GPT-NeoX is the third step towards a GPT-3 size open model by EleutherAI.
  It was the largest open model at the time of its release (20B), but far from the size of GPT-3 (175B).

  It was announced in Feb 2022: https://blog.eleuther.ai/announcing-20b/
  Publication: https://arxiv.org/abs/2204.06745

  Differences to GPT-2:
    - Rotary Position Embedding (RoPE)
    - Untied embedding / un-embedding matrices (Separate parameters for token embeddings and generating logits.)
    - Uses faster GELU approximation as activation function
    - The weights are stored in vertical matrices (easier to execute the vector-matrix multiplication)
    - 16 bit parameters (FLOAT16)

  Differences to GPT-J:
    - Original structure of residual connections (the parallelized attention and feed forward technique is dropped)
    - Biases are used
    - Attention scale is used
    - Neural net normalization weights are different per decoders (not common, as at GPT-J)
    - The rotary position embedding is applied in sliced arrangement (not interleaved)
    - Uses faster GELU approximation as activation function
    - 16 bit parameters (FLOAT16)

  @author Hunor Szegi
 */
public class GPTNeoX extends BaseTransformer
{
    Parameter tokenEmbeddings, embeddingWeight, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "gpt_neox.embed_in.weight",         tokenCount, hiddenSize);
        embeddingWeight = loadMatrix(EMBEDDING,   "embed_out.weight",                 tokenCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "gpt_neox.final_layer_norm.weight", hiddenSize);
        normBias        = loadVector(NORM_BIAS,   "gpt_neox.final_layer_norm.bias",   hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        return matrix(tokenEmbeddings).row(tokenId);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(embeddingWeight));

        return selectBestToken(logits, topK);
    }
}
