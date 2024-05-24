package huplay.transformer._2022_05_big_science_bloom;

import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.transformer.TransformerUtil.layerNorm;
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
    // TODO: Maybe something isn't perfect here, the output looks good, but very ofter repeats itself.
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, "word_embeddings.weight", tokenCount, hiddenSize);
        loadVector(INPUT_NORM_WEIGHT, "word_embeddings_layernorm.weight", hiddenSize);
        loadVector(INPUT_NORM_BIAS, "word_embeddings_layernorm.bias", hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getVector(token);

        // Input normalization
        return layerNorm(hiddenState, vector(INPUT_NORM_WEIGHT), vector(INPUT_NORM_BIAS), epsilon);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);

        return determineOutputToken(hiddenState, topK);
    }
}
