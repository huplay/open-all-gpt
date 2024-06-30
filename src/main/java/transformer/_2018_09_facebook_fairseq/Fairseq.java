package transformer._2018_09_facebook_fairseq;

import config.Parameter;
import math.dataType.vector.Vector;
import position.sinusoid.SinusoidPositionEmbedding;
import transformer.BaseTransformer;

import static config.ParameterType.*;
import static math.BasicMathUtility.sqrt;
import static math.MathUtil.MATH;

/**
  Fairseq transformer

  FAIR (Fundamental AI Research) was created by Facebook as its AI research department. (Renamed to Meta AI in 2021.)
  FAIR developed the PyTorch framework, and the Fairseq toolkit (FAIR Sequence-to-Sequence) for training models.

  Fairseq supports a wide range of architectures; the Transformer implementation was added in June 2018.
  The code of the Fairseq framework: https://github.com/facebookresearch/fairseq
  Changelog of the 0.5.0 release: https://github.com/facebookresearch/fairseq/releases/tag/v0.5.0
  Official page: https://ai.meta.com/tools/fairseq/
  Publication: https://arxiv.org/abs/1904.01038

  The first published decoder-only model using Fairseq: Baevski and Auli, 28 Sep 2018: https://arxiv.org/abs/1809.10853
  Models (in pytorch format): https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md

  Other notable use case was the XGML (Presumably Cross-lingual Generative Language Model) published in Dec 2021.
  Publication: https://arxiv.org/abs/2112.10668
  Code and parameters (GitHub): https://github.com/facebookresearch/fairseq/tree/main/examples/xglm
  Parameters (Hugging Face, in safetensors format as well): https://huggingface.co/facebook

  On the same day the same team published another research, comparing vanilla to mixture of experts models,
  where they trained 6 vanilla (125M, 355M, 1.3B, 2.7B, 6.7B, 13B) and 4 mixture of experts models.
  (In this context the "dense" is a vanilla transformer, and "sparse" is a mixture of experts model.)
  These models weren't named, and the XGLM paper wasn't referenced, but the used arhitecture is identical.
  At that time the 13B model was the largest open model available for researchers I know about.
  Publication: https://arxiv.org/abs/2112.10684
  Models (in pytorch format): https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
  These models are not uploaded to the Hugging Face platform officially, but others converted it to safetensors format:
  https://huggingface.co/KoboldAI

  Fairseq was used for many research projects, and during the years its Transformer implementation was enhanced.
  In the initial implementation they used ReLU activation function and the query/key/value matrix was merged.
  In th v0.9.0 version (Dec 2019) the matrices were separated, and the newer models preferred the GELU activation.
  Beside the sinusoidal position embedding the learned is also supported.

  This implementation supports the earlier versions, with sinusoidal position embeddings.
  The query/key/value split and the activation function can be configured,
  so the Baevski-Auli, and the XGML models can be used as well.
  (In the Hugging Face platform few Fairseq models are implemented separately, e.g. XGML, and encoder-decoder cases also.)

  Differences to GPT-2:
    - Sinusoidal position embedding
    - The attention scale is applied on the query, not on the score
    - The weights are stored in vertical matrices (easier to execute the vector-matrix multiplication)
    - The position embedding matrix contains 2 extra rows (not used at inference, but the position index should be adjusted)
    - The query/key/value matrices are stored separately (in later versions, like XGML)
    - 16 bit parameters (FLOAT16) (at later versions, like XGML)

 * @author Hunor Szegi
 */
public class Fairseq extends BaseTransformer
{
    Parameter tokenEmbeddings, embeddingWeight, normWeight, normBias;
    SinusoidPositionEmbedding position = new SinusoidPositionEmbedding();

    float embeddingScale;

    public void loadParameters()
    {
        tokenEmbeddings = loadMatrix(EMBEDDING,   "model.embed_tokens.weight", tokenCount, hiddenSize);
        embeddingWeight = loadMatrix(EMBEDDING,   "lm_head.weight",            tokenCount, hiddenSize);
        normWeight      = loadVector(NORM_WEIGHT, "model.layer_norm.weight",   hiddenSize);
        normBias        = loadVector(NORM_BIAS,   "model.layer_norm.bias",     hiddenSize);

        position.initSliced(config, hiddenSize);

        embeddingScale = sqrt(hiddenSize);
    }

    public Vector preProcessToken(int pos, int tokenId)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(tokenEmbeddings).row(tokenId);

        hiddenState = hiddenState.multiply(embeddingScale);

        // Position embedding
        hiddenState = position.apply(hiddenState, pos + 2);

        return hiddenState;
    }

    public int generateToken(Vector hiddenState, int topK)
    {
        // Final normalization
        hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        Vector logits = hiddenState.multiplyByTransposed(matrix(embeddingWeight));

        return selectBestToken(logits, topK);
    }
}
