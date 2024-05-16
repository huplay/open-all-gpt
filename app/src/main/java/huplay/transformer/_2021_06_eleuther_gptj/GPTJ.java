package huplay.transformer._2021_06_eleuther_gptj;

import huplay.transformer.BaseTransformer;
import huplay.util.Vector;

import static huplay.transformer.TransformerUtil.layerNorm;
import static huplay.config.ParameterType.*;

/**
 * TODO:
 * https://github.com/jzhang38/TinyLlama/issues/24
 *
 * GPTJ style: Original Llama, llama.cpp
 * rotates pairs of even and odd dimensions
 *
 * NEOX style: OpenLlama (all HF Llama)
 * rotates the 1st and 2nd half
 *
 * HF permutes the weight

  EleutherAI GPT-J transformer

  Differences to GPT-NEO:
    - Rotary Position Embedding (RoPE)
    - Uses bias at token embeddings
    - No bias at attention query/key/value matrices and projection (but has bias at the mlp component)
    - Feed-forward normalization parameters are common in all decoders, and the same used at final normalization
 * @author Hunor Szegi
 */
public class GPTJ extends BaseTransformer
{
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, "lm_head.weight", tokenCount, hiddenSize);
        loadVector(TOKEN_EMBEDDING_BIAS, "lm_head.bias", tokenCount); // TODO: This is new

        loadVector(OUTPUT_NORM_WEIGHT, "transformer.ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "transformer.ln_f.bias", hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        return matrix(TOKEN_EMBEDDINGS)[token];
        //hiddenState = UTIL.addVectors(hiddenState, vector(TOKEN_EMBEDDING_BIAS));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);

        return determineOutputToken(hiddenState, topK);
    }
}
