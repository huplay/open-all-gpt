# Google Brain, the original decoder-only Transformer #

The encoder-decoder architecture was described in the famous "Attention Is All You Need" paper:
June 2017: https://arxiv.org/abs/1706.03762

The decoder-only variant was described in "Generating Wikipedia by Summarizing Long Sequences"
Jan 2018: https://arxiv.org/abs/1801.10198

Features:
- Sinusoid position embedding added to the input at the beginning
(But they created a variant of the original encoder-decoder architecture using learned position embedding as well.)
- Normalization is used at the end of the attention and feed-forward blocks
- Residual connections at the attention and feed-forward blocks
- Multi-head attention
- Scale the attention score by 1 / sqrt(headSize)
- Single layer projection at the end of the attention blocks
- Feed-forward block has two layers (layer1: 4 * hiddenSize neurons, layer2: hiddenSize neurons)
- ReLU activation function (used only at the first feed-forward layer)
- 32 bit parameters
- query/key/value matrices are stored in a single matrix

The weights of the trained model weren't published.
I don't know about any model which uses exactly the same architecture,
so currently you can't try this implementation.