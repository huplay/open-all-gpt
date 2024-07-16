# Google Gemma #

Gemma was announced and released on 21 Feb 2024. Publication: https://arxiv.org/abs/2403.08295

The architecture is almost identical to the Llama models, even the parameter names are the same.

Differences to Llama:
- Llama 2 introduced the GQA attention, where the same key/value is used at multiple queries (Grouped Query).
Gemma 2B is an edge case of the GQA, where the same key/value is used at all queries. (MQA: Multi Query)
- Extra input normalization: division by sqrt of the hidden size
- The RSM normalization adds 1 to the weight (bias)
- GELU activation function (same as at GPT-2 and most of the models)
- The rotary position embedding is applied in sliced arrangement (not interleaved) (same as at EleutherAI GPT-NeoX)
