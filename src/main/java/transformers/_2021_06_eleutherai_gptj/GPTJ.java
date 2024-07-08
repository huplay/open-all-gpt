package transformers._2021_06_eleutherai_gptj;

import config.Parameter;
import transformer.serial.BaseTransformer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
  EleutherAI GPT-J transformer

  GPT-J was the second step towards a GPT-3 size open model by EleutherAI and Ben Wang (Mesh-Transformer-JAX).
  It was the largest open model at that time (6B), but far from the size of GPT-3 (175B).
  It was the first model using the RoPE position embedding.
  Source code: https://github.com/kingoflolz/mesh-transformer-jax

  Differences to GPT-NEO:
    - Rotary Position Embedding (RoPE)
    - Parallelized attention and feed-forward technique:
        The attention block and the neural net block have the same input (it would be possible to execute parallel).
        At other transformers, the output of the attention block is the input of the neural net block;
        and there are separate residual connections within the attention block and within the neural net block.
        Here, there's a single residual connection, joining the attention input and the neural net output.
        (More precisely, the join starts within the attention block, between the normalization and attention mechanism.)
    - Untied embedding / un-embedding matrices (Separate parameters for token embeddings and generating logits.)
    - No bias at attention query/key/value matrices and projection (but has bias at the neural net component)
    - Neural net normalization parameters are common in all decoders, and the same used at final normalization
    - The weights are stored in vertical matrices (easier to execute the vector-matrix multiplication)

  @author Hunor Szegi
 */
public class GPTJ extends BaseTransformer
{
    Parameter tokenEmbeddings, embeddingWeight, embeddingBias, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,      "transformer.wte.weight",  tokenCount, hiddenSize);
        embeddingWeight = loadMatrix(EMBEDDING,      "lm_head.weight",          tokenCount, hiddenSize);
        embeddingBias   = loadVector(EMBEDDING_BIAS, "lm_head.bias",            tokenCount);
        normWeight      = loadVector(NORM_WEIGHT,    "transformer.ln_f.weight", hiddenSize);
        normBias        = loadVector(NORM_BIAS,      "transformer.ln_f.bias",   hiddenSize);
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
        logits = logits.add(vector(embeddingBias));

        return selectBestToken(logits, topK);
    }
}
