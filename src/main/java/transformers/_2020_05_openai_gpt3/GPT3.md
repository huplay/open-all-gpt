# OpenAI GPT-3 #

After the completely published and downloadable GPT and GPT-2 models (up to 1.5B) OpenAI created GPT-3,
but without sharing the trained parameters. The largest variant has 175B parameters.

GPT-3 was announced on 28 May 2020 by this publication: https://arxiv.org/abs/2005.14165
Restricted (beta) access was announced on 11 June 2020, but only through API, the model is running on remote servers.
Since 18 Nov 2021 the API is publicly available.

Later, newly trained variants were created, branded as GPT 3.5.
On 30 Nov 2022 the latest fine-tuned 3.5 version was announced as ChatGPT.

Difference to GPT-2:
- Sparse attention: Every second decoder has local attention, using only the last 256 tokens

There are no available GPT-3 models, but this implementation works with any GPT-2 like models. (Including Cerebras.)
The result is identical to the GPT-2 transformer until the 256 token context size is reached,
and it should work with longer text as well, just the result will be slightly worse.