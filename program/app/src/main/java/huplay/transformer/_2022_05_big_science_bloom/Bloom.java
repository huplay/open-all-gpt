package huplay.transformer._2022_05_big_science_bloom;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
import static huplay.config.ParameterType.*;

/**
  BLOOM transformer

  Differences to GPT-2:
    - ALiBi position embedding, applied in the attention block to the score
    - 16 bit parameters (BFLOAT16 for the 176B model, FLOAT16 for the others)
    - Additional normalization at input (it was necessary because of the FLOAT16 data type, but used at all models)
    - The weights are stored in transposed matrices (easier to execute the vector-matrix multiplication)
    - The values in the query/key/value matrices are ordered first by head, and second by type
    - The key and value vectors are stored separately for the heads

 * @author Hunor Szegi
 */
public class Bloom extends BaseTransformer
{
    Parameter tokenEmbeddings, inputNormWeight, inputNormBias, outputNormWeight, outputNormBias;

    public void loadParameters()
    {
        tokenEmbeddings  = loadMatrix(EMBEDDING,   "word_embeddings.weight",           tokenCount, hiddenSize);
        inputNormWeight  = loadVector(NORM_WEIGHT, "word_embeddings_layernorm.weight", hiddenSize);
        inputNormBias    = loadVector(NORM_BIAS,   "word_embeddings_layernorm.bias",   hiddenSize);
        outputNormWeight = loadVector(NORM_WEIGHT, "ln_f.weight",                      hiddenSize);
        outputNormBias   = loadVector(NORM_BIAS,   "ln_f.bias",                        hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).getRow(token);

        // Input normalization
        return MATH.layerNorm(hiddenState, vector(inputNormWeight), vector(inputNormBias), epsilon);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(outputNormWeight), vector(outputNormBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(tokenEmbeddings)).getValues();

        return selectBestToken(logits, topK);
    }
}
