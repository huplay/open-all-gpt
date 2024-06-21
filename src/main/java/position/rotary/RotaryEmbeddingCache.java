package position.rotary;

import java.util.Map;

/**
 * Holder of cos and sin embedding matrices
 */
record RotaryEmbeddingCache(float[][] cos, float[][] sin)
{
    // Cache map, to be able to store the matrices for multiple models (don't expect it will be used, but I can support)
    // It's static, so this cache will be shared by multiple instances (all decoder layers will reach the same)
    static Map<String, RotaryEmbeddingCache> ROTARY_EMBEDDING_CACHE;
}
