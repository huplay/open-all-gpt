package transformers._2019_02_openai_gpt2.parallel;

import config.Parameter;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.parallel.ParallelBaseTransformer;

import java.util.List;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
 * OpenAI GPT-2 transformer (Parallel implementation)
 * @author Hunor Szegi
 */
public class ParallelGPT2 extends ParallelBaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings    = loadMatrix(EMBEDDING,   "wte.weight",  tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING,   "wpe.weight",  contextSize, hiddenSize);
        normWeight         = loadVector(NORM_WEIGHT, "ln_f.weight", hiddenSize);
        normBias           = loadVector(NORM_BIAS,   "ln_f.bias",   hiddenSize);
    }

    public Matrix preProcessInputTokens(int posOffset, List<Integer> tokenIds)
    {
        Matrix hiddenState = emptyMatrix(tokenIds.size(), hiddenSize);

        for (var pos = 0; pos < tokenIds.size(); pos++)
        {
            // Find the embeddings of the token (this is the initial hidden state)
            Vector embedding = matrix(tokenEmbeddings).row(tokenIds.get(pos));

            // Find the position embedding of the position
            // posOffset > 0 if this is a subsequent input
            Vector positionEmbedding = matrix(positionEmbeddings).row(pos + posOffset);

            // The hidden state is the sum of the token embedding and the position embedding
            hiddenState.setRow(pos, embedding.add(positionEmbedding));
        }

        return hiddenState;
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
