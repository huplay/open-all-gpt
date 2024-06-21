package position.rotary;

import config.Config;
import math.dataType.vector.Vector;

import java.util.HashMap;

import static math.BasicMathUtility.*;
import static position.rotary.RotaryEmbeddingCache.ROTARY_EMBEDDING_CACHE;

/**
  Rotary position embedding (RoPE)

  Original publication:
    - Code (Zhuiyi Technology, first commit in 22 Mar 2021): https://github.com/ZhuiyiTechnology/roformer
    - Blog post (By Su Jianlin, Zhuiyi Technology, in Chinese, 23 Mar 2021): https://kexue.fm/archives/8265
    - Paper in English (RoFormer) (20 Apr 2021, Jianlin Su et al., Zhuiyi Technology): https://arxiv.org/abs/2104.09864

  Implementations in frameworks:
    - Mesh Transformer JAX (by Ben Wang):
      Commit "add rotary pos encoding, make pos encoding configurable" (18 Apr, 2021):
      https://github.com/kingoflolz/mesh-transformer-jax/commit/728d20b26785495852c8851ce0d770bc10d3caf0
    - Pytorch (29 Jun - 16 Aug 2021): https://github.com/lucidrains/rotary-embedding-torch
    - Hugging Face Transformers (31 Aug 2021):
      https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gptj/modeling_gptj.py

  Usage in models:
    - EleutherAI GPT-J6B (Released in June 2021, using the Mesh Transformer JAX.)
    - EleutherAI Neo-X (Released in Feb 2022)
    - Meta Llama (Released in Feb 2023)
    - Google Gemini, Gemma (Released in Feb 2024)

  Blog posts:
    - EleutherAI: https://blog.eleuther.ai/rotary-embeddings/
    - Hugging Face documentation (RoFormer): https://huggingface.co/docs/transformers/model_doc/roformer
 */
public class RotaryPositionEmbedding
{
    // A cache key is necessary to be able to use multiple models parallel
    private final String cacheKey;

    /**
     * Create a new rotary embedder, initializing (and caching) the cos and sin matrices to be used later
     */
    public RotaryPositionEmbedding(Config config, int vectorSize)
    {
        if (ROTARY_EMBEDDING_CACHE == null)
        {
            // Create a new cache instance if it is the first usage
            ROTARY_EMBEDDING_CACHE = new HashMap<>();
        }

        // The cache key will be the model path to make sure this is unique for a model
        cacheKey = config.getModelPath();
        if (!ROTARY_EMBEDDING_CACHE.containsKey(cacheKey))
        {
            // Initialize the embedding matrices if it is the first usage of this model
            var contextSize = config.getContextSize();
            int halfSize = vectorSize / 2;

            var cos = new float[contextSize][halfSize];
            var sin = new float[contextSize][halfSize];

            for (int pos = 0; pos < contextSize; pos++)
            {
                for (int i = 0; i < halfSize; i++)
                {
                    var degree = pos / pow(10000, (float) i / halfSize);
                    cos[pos][i] = cos(degree);
                    sin[pos][i] = sin(degree);
                }
            }

            ROTARY_EMBEDDING_CACHE.put(cacheKey, new RotaryEmbeddingCache(cos, sin));
        }
    }

    /**
     * Position embedding, using the original, interleaved organisation ("rotate_every_two")
     * (Organised as: a1, b1, a2, b2, .... an, bn)
     * Used in: EleutherAI GPT-J, Meta Llama
     * (Comment on Llama:
     *   The original Llama implementation is interleaved.
     *   The Hugging Face Llama implementation converts the weights to sliced (using permute),
     *   and after that a sliced algorithm is used, but overall this is equivalent with the interleaved.)
     *
     */
    public void applyInterleaved(Vector vector, int pos)
    {
        // Read the initialized cos and sin matrices
        RotaryEmbeddingCache cachedRotaryEmbeddings = ROTARY_EMBEDDING_CACHE.get(cacheKey);
        var cos = cachedRotaryEmbeddings.cos();
        var sin = cachedRotaryEmbeddings.sin();

        int halfSize = vector.size() / 2;

        // Apply the rotation on every two values (iterating over on the vector, stepping by 2)
        for (int i = 0; i < halfSize; i++)
        {
            var n = 2 * i;

            // Get the value of the two
            float a = vector.get(n);
            float b = vector.get(n + 1);

            // Apply the rotation
            vector.set(n    , a * cos[pos][i] - b * sin[pos][i]);
            vector.set(n + 1, b * cos[pos][i] - a * sin[pos][i]);
        }
    }

    /**
     * Position embedding, using sliced organisation ("rotate_half")
     * (Organised as: a1, a2, ... an, b1, b2, ... bn)
     * Used in: EleutherAI NeoX, Google Gemini, Gemma
     */
    public void applySliced(Vector vector, int pos)
    {
        // Read the initialized cos and sin matrices
        RotaryEmbeddingCache cachedRotaryEmbeddings = ROTARY_EMBEDDING_CACHE.get(cacheKey);
        var cos = cachedRotaryEmbeddings.cos();
        var sin = cachedRotaryEmbeddings.sin();

        int halfSize = vector.size() / 2;

        // Apply the rotation on every value (iterating over on the vector, stepping by 2)
        for (int i = 0; i < halfSize; i++)
        {
            // Get the value of the two
            float a = vector.get(i);
            float b = vector.get(i + halfSize);

            // Apply the rotation
            vector.set(i           , a * cos[pos][i] - b * sin[pos][i]);
            vector.set(i + halfSize, b * cos[pos][i] - a * sin[pos][i]);
        }
    }
}
