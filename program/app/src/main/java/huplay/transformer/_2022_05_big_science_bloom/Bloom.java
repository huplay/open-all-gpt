package huplay.transformer._2022_05_big_science_bloom;

import huplay.config.Parameter;
import huplay.transformer.BaseTransformer;
import huplay.dataType.vector.Vector;

import static huplay.MathUtilProvider.MATH;
import static huplay.config.Parameter.par;
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
    // Declare the used parameters (id, parameter type):
    Parameter TOKEN_EMBEDDINGS = par("word_embeddings.weight", EMBEDDINGS);
    Parameter INPUT_NORM_WEIGHT = par("word_embeddings_layernorm.weight", NORMALIZATION_WEIGHT);
    Parameter INPUT_NORM_BIAS = par("word_embeddings_layernorm.bias", NORMALIZATION_BIAS);
    Parameter OUTPUT_NORM_WEIGHT = par("ln_f.weight", NORMALIZATION_WEIGHT);
    Parameter OUTPUT_NORM_BIAS = par("ln_f.bias", NORMALIZATION_BIAS);

    // TODO: Maybe something isn't perfect here, the output looks good, but very ofter repeats itself.
    public void loadParameters()
    {
        loadMatrix(TOKEN_EMBEDDINGS, tokenCount, hiddenSize);
        loadVector(INPUT_NORM_WEIGHT, hiddenSize);
        loadVector(INPUT_NORM_BIAS, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, hiddenSize);
    }

    public Vector preProcessToken(int pos, int token)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS).getRow(token);

        // Input normalization
        return MATH.layerNorm(hiddenState, vector(INPUT_NORM_WEIGHT), vector(INPUT_NORM_BIAS), epsilon);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = MATH.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getValues();

        return selectBestToken(logits, topK);
    }
}
