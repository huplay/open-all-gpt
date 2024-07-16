package transformers._2024_02_google_gemma.parallel;

import config.Parameter;
import math.dataType.BrainFloat16;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.parallel.ParallelBaseTransformer;

import java.util.List;

import static config.ParameterType.EMBEDDING;
import static config.ParameterType.NORM_WEIGHT;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
 * Google Gemma transformer (Parallel implementation)
 * @author Hunor Szegi
 */
public class ParallelGemma extends ParallelBaseTransformer
{
    Parameter tokenEmbeddings, normWeight;

    float normConstant;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "embed_tokens.weight", tokenCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "norm.weight",         hiddenSize);

        // Constant for extra input normalization (downcast to BrainFloat16 and back to Float32)
        normConstant = BrainFloat16.of(sqrt(hiddenSize)).getFloatValue();
    }

    public Matrix preProcessInputTokens(int posOffset, List<Integer> tokenIds)
    {
        Matrix embeddings = matrix(tokenEmbeddings);
        Matrix hiddenState = Matrix.emptyMatrix(DataType.FLOAT_32, tokenIds.size(), hiddenSize);

        for (var pos = 0; pos < tokenIds.size(); pos++)
        {
            // Find the embeddings of the token (this is the initial hidden state) // TODO: Wrong matrix type. Matrix type vs lines
            Vector embedding = Vector.of(DataType.FLOAT_32, embeddings.row(tokenIds.get(pos)).getValues());
            embedding = embedding.multiply(normConstant);
            hiddenState.setRow(pos, embedding);
        }

        return hiddenState;
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        hiddenState = Vector.of(DataType.FLOAT_32, hiddenState.getValues());

        // Extra input normalization: division by sqrt of the hidden size
        return hiddenState.multiply(normConstant);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.RMSLayerNorm(hiddenState, vector(normWeight), epsilon, 1f);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
