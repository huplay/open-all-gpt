package transformer._2022_05_meta_opt;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.BaseTransformer;

import static config.ParameterType.*;
import static math.MathUtil.MATH;

/**
  Meta (Facebook) OPT transformer (Open Pre-Trained Transformer)

  Meta trained 9 models and made it accessible for research (non-commercial use) on 3 May 2022.
  The largest model has an equivalent size to GPT-3. (It was the largest open model at the time.)
  (The 66B model was released only on 22 June 2022.)
  Publication: https://arxiv.org/abs/2205.01068

  Differences to GPT-2:
    - ReLU activation function (not GELU)
    - The attention dividend is applied on the query, not on the score
    - The weights are stored in vertical matrices (easier to execute the vector-matrix multiplication)
    - The query/key/value matrices are stored separately
    - The position embedding matrix contains 2 extra rows (not used at inference, but the position index should be adjusted)
    - 16 bit parameters (FLOAT16)
    - The OPT-350M model is different to the others, so it is implemented separately (see OPT350)


 : it performs the normalization after the attention and neural net blocks,
      not before, and the final normalization is omitted. (Same as at GPT-1)

  @author Hunor Szegi
 */
public class OPT extends BaseTransformer
{
    public Parameter tokenEmbeddings, positionEmbeddings, normWeight, normBias;

    public void loadParameters()
    {
        tokenEmbeddings    = loadMatrix(EMBEDDING,   "embed_tokens.weight",     tokenCount, hiddenSize);
        positionEmbeddings = loadMatrix(EMBEDDING,   "embed_positions.weight",  contextSize + 2, hiddenSize);
        normWeight         = loadVector(NORM_WEIGHT, "final_layer_norm.weight", hiddenSize);
        normBias           = loadVector(NORM_BIAS,   "final_layer_norm.bias",   hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token (this is the initial hidden state)
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Find the position embedding of the position
        Vector positionEmbedding = matrix(positionEmbeddings).row(pos + 2);

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
