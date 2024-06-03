package huplay.transformer._2021_06_eleuther_gptj;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
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
    Parameter TOKEN_EMBEDDINGS, TOKEN_EMBEDDING_BIAS, NORM_WEIGHT, NORM_BIAS;

    public void loadParameters()
    {
        TOKEN_EMBEDDINGS = loadMatrix("lm_head.weight", EMBEDDINGS, tokenCount, hiddenSize);
        TOKEN_EMBEDDING_BIAS = loadVector("lm_head.bias", EMBEDDINGS_BIAS, tokenCount);
        NORM_WEIGHT = loadVector("transformer.ln_f.weight", NORMALIZATION_WEIGHT, hiddenSize);
        NORM_BIAS = loadVector("transformer.ln_f.bias", NORMALIZATION_BIAS, hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getRow(token);
        return MATH.addVectors(hiddenState, vector(TOKEN_EMBEDDING_BIAS));
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(NORM_WEIGHT), vector(NORM_BIAS), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }
}
