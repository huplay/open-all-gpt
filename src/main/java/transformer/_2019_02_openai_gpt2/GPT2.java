package transformer._2019_02_openai_gpt2;

import config.Parameter;
import transformer.BaseTransformer;
import math.dataType.vector.Vector;

import static math.MathUtil.MATH;
import static config.ParameterType.*;

/**
  OpenAI GPT-2 transformer

  It was announced and partly released on 14 Feb 2019, and fully released on 5 Nov 2019.
  Source code: https://github.com/openai/gpt-2
  Publication: https://paperswithcode.com/paper/language-models-are-unsupervised-multitask

  Differences to GPT-1:
    - The normalization is used at the beginning of the attention and mlp blocks
    - Final normalization is added after the last decoder

  (The normalization before the first decoder's attention block gives more numerical stability at larger models.)

  @author Hunor Szegi
 */
public class GPT2 extends BaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings    = loadMatrix(EMBEDDING,   "wte.weight",  tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING,   "wpe.weight",  contextSize, hiddenSize);
        normWeight         = loadVector(NORM_WEIGHT, "ln_f.weight", hiddenSize);
        normBias           = loadVector(NORM_BIAS,   "ln_f.bias",   hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token (this is the initial hidden state)
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Find the position embedding of the position
        Vector positionEmbedding = matrix(positionEmbeddings).row(pos);

        // Return the addition of the hidden state and the position embedding
        return hiddenState.add(positionEmbedding);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
