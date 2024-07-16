# GPT-2 #

OpenAI's GPT-2 was announced and partly released on 14 Feb 2019, and fully released on 5 Nov 2019.

- Source code: https://github.com/openai/gpt-2
- Publication: https://paperswithcode.com/paper/language-models-are-unsupervised-multitask

Differences to GPT-1:
- Pre-normalization (instead of post-normalization)
  - The normalization is used at the beginning of the attention and mlp blocks
  - Final normalization is added after the last decoder
  - The residual connection placed differently

The pre-normalization gives more numerical stability. Firstly, because this is an extra normalization before the first decoder (after the position embedding).
Secondly, because the residual connection has changed also.

GPT-1: [Emb] [Pos] [Att-Add-Norm] [MLP-Add-Norm] ... [Att-Add-Norm] [MLP-Add-Norm] [Logits]

GPT-2: [Emb] [Pos] [Norm-Att-Add] [Norm-MLP-Add] ... [Norm-Att-Add] [Norm-MLP-Add] [Final norm] [Logits]

At GPT-1 there is a normalization right before every attention, MLP and Logits, except before the first attention.
Adding the extra normalization there would result the following:

[Emb] [Pos] [Norm] [Att-Add-Norm] [MLP-Add-Norm] ... [Att-Add-Norm] [MLP-Add-Norm] [Logits]

But we have one more difference here, because the residual connection (Add) uses the value before the normalization.
That's why it is grouped differently.

Earlier, the residual connection was more separated from the input. The input went into the attention block, added to the result, and the normalization washed both.

Now, the residual connection uses the value before the normalization. So the input has a bigger effect on the deeper layers. But it's still stable numerically, because there's a normalization right before every attention.
