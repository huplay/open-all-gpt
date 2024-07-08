package transformers._2022_05_meta_opt.opt350;

import config.Parameter;
import math.dataType.vector.Vector;
import transformer.serial.BaseTransformer;

import static config.ParameterType.*;

/**
  Meta (Facebook) OPT-350 transformer (Open Pre-Trained Transformer)

  Meta trained 9 models and made it accessible for research (non-commercial) on 3 May 2022.
  Interestingly, the second model is very different to the others architecturally, so it is implemented here separately.

  The core Transformer class is completely rewritten.
  The attention and neural net layer classes are extended from the standard OPT implementation,
  so only one method is overwritten in both, the rest is inherited.

  Differences to the other OPT models:
    - Performs the normalization after the attention and neural net blocks (not before),
      and because of that the final normalization is omitted. (Same as at GPT-1)
    - The record size of the word embedding matrix is only 512, not matching to the hidden size (1024).
      Extra project in/out parameters (weights) are used to convert the 512 embedding to the size of 1024 and back

  I guess the 350M model was created first, and they changed the structure for the larger models.
  (The pre-normalization provides more numerical stability. (Important for larger models. They learned it meantime.)
  And most likely they decided later to create the smallest (125M) model, which also uses the newer architecture.

  The prefix of the parameters is also different, the "model." part is missing. (But it's common with the 6.7B model.)

  @author Hunor Szegi
 */
public class OPT350 extends BaseTransformer
{
    Parameter tokenEmbeddings, positionEmbeddings, projectIn, projectOut;

    public void loadParameters()
    {
        // It is an OPT 350M specific settings, so in practice fixed, but we read it from the config, defaulting to 512
        int projectionSize = config.getIntOptional("word_embed_proj_dim", 512);

        tokenEmbeddings    = loadMatrix(EMBEDDING,       "embed_tokens.weight",    tokenCount, projectionSize);
        positionEmbeddings = loadMatrix(EMBEDDING,       "embed_positions.weight", contextSize + 2, hiddenSize);
        projectIn          = loadMatrix(VERTICAL_WEIGHT, "project_in.weight",      hiddenSize, projectionSize);
        projectOut         = loadMatrix(VERTICAL_WEIGHT, "project_out.weight",     projectionSize, hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token (this is the initial hidden state)
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        // Extra projection step (OPT 350M specific)
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectIn));

        // Find the position embedding of the position
        Vector positionEmbedding = matrix(positionEmbeddings).row(pos + 2);

        // Return the addition of the hidden state and the position embedding
        return hiddenState.add(positionEmbedding);
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Extra projection step (OPT 350M specific)
        hiddenState = hiddenState.multiplyByTransposed(matrix(projectOut));

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(tokenEmbeddings));

        return selectBestToken(logits, topK);
    }
}
