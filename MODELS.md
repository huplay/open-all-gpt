## Models ##

### June 2017 - The original Google Brain Transformer ###

The first Transformer published in the famous "Attention Is All You Need" paper was an encoder-decoder architecture for translation. This app supports only the decoder-only architectures, but for comparison I added the details here.
They trained multiple variants, the two most important listed in the table. The parameters have not been published.

| Name             | Hidden size | Enc. no. | Dec. no. | Head no. | Max. length | Size of params |
|------------------|------------:|---------:|---------:|---------:|------------:|---------------:|
| Transformer Base |         512 |        6 |        6 |        8 |           ? |           65 M |
| Transformer Big  |        1024 |        6 |        6 |       16 |           ? |          213 M |

## Jan 2018 - Decoder-only Transformer ###

Half a year later the Google Brain team published a decoder-only Transformer architecture. Not all the details are published, but it is known their best model had 5 decoder layers, alternating between local and global attention. They tried mixture of experts variants as well. The parameters have not been published.

| Name                     | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------------------|------------:|---------:|---------:|------------:|---------------:|
| Decoder-only Transformer |           ? |        5 |        ? |           ? |            ? M |

### Jun 2018 - GPT-1 (OpenAI) ###

OpenAI was the first to recreate the decoder-only Transformer architecture, and they published the model completely. (Code and trained parameters as well.) You can try it using the app:

| Name  | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-------|------------:|---------:|---------:|------------:|---------------:|
| GPT-1 |         768 |       12 |       12 |        1024 |          124 M |

### Sep 2018 - Fairseq (FAIR, Meta AI) ###

Fairseq toolkit was created by FAIR (later Meta AI). Transformer support was added in June 2018, first published decoder-only usage by Baevski-Auli. (Adaptive input)

| Name                        | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------------------------|------------:|---------:|---------:|------------:|---------------:|
| Adaptive input (very large) |        1024 |       16 |       16 |        1024 |         1026 M |

### Feb 2019 - GPT-2 (OpenAI) ###

The GPT-2 models were published completely. (Code and trained parameters as well.) You can try all of these using the app:

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-2 SMALL  |         768 |       12 |       12 |        1024 |          124 M |
| GPT-2 MEDIUM |        1024 |       24 |       16 |        1024 |          355 M |
| GPT-2 LARGE  |        1280 |       36 |       20 |        1024 |          774 M |
| GPT-2 XL     |        1600 |       48 |       25 |        1024 |        1,558 M |

### May 2020 - GPT-3 (OpenAI) ###

The GPT-3 algorithm is known (almost identical to GPT-2), this application has implemented it, but the parameters are not published, so you can't use these here:

| Name                       | Hidden size | Dec. no. | Head no. | Max. length |   Size of params | 
|----------------------------|------------:|---------:|---------:|------------:|-----------------:|
| GPT-3 SMALL                |         768 |       12 |       12 |        2048 |            124 M |
| GPT-3 MEDIUM               |        1024 |       24 |       16 |        2048 |            355 M |
| GPT-3 LARGE                |        1536 |       24 |       16 |        2048 |            759 M |
| GPT-3 XL                   |        2048 |       24 |       24 |        2048 |          1,314 M |
| GPT-3 ADA                  |        2560 |       32 |       32 |        2048 |          2,649 M |
| GPT-3 BABBAGE              |        4096 |       32 |       32 |        2048 |          6,654 M |
| GPT-3 CURIE                |        5140 |       40 |       40 |        2048 |         12,948 M |
| GPT-3 DAVINCI / GPT-3      |       12288 |       96 |       96 |        2048 |        174,591 M |
| GPT-3 DAVINCI v2 / GPT-3.5 |       12288 |       96 |       96 |        4000 |        174,591 M |
| GPT-3 DAVINCI v3 / ChatGPT |       12288 |       96 |       96 |        4000 |        174,591 M |

### March 2021 - GPT-Neo (EleutherAI) ###

`EleutherAI` attempted to recreate the GPT-3 models to make it accessible for everyone. Initially it resulted the GPT-Neo series.

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-Neo 125M |         768 |       12 |       12 |        2048 |          124 M |
| GPT-Neo 350M |        1024 |       24 |       16 |        2048 |          355 M |
| GPT-Neo 1.3B |        2048 |       24 |       16 |        2048 |        1,314 M |
| GPT-Neo 2.7B |        2560 |       32 |       20 |        2048 |        2,649 M |

### June 2021 - GPT-J (EleutherAI) ###

GPT-J was the second step towards a GPT-3 size open model by `EleutherAI` and Ben Wang (Mesh-Transformer-JAX).
It was the largest open model at that time (6B), but far from the size of GPT-3 (175B).
It was the first model using the RoPE position embedding.

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-J-6B     |        4096 |       28 |       16 |        2048 |        5,849 M |

### XGLM (FAIR, Meta AI)

XGML (Presumably Cross-lingual Generative Language Model) was published in Dec 2021. These are models trained on multi-language corpus.

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Comment     |
|-----------|------------:|---------:|---------:|------------:|-------------|
| XGLM 564M |        1024 |       24 |       16 |        2048 |             |
| XGLM 1.7B |        2048 |       24 |       16 |        2048 |             |
| XGLM 2.9B |        2048 |       48 |       16 |        2048 |             |
| XGLM 4.5B |        2048 |       48 |       16 |        2048 | Added later |
| XGLM 7.5B |        4096 |       32 |       32 |        2048 |             |

On the same day the same team published another research, comparing vanilla to mixture of experts models,
where they trained 6 vanilla and 4 mixture of experts models. (In this context the "dense" is a vanilla transformers, and "sparse" is a mixture of experts model.)

(This app supports only the vanilla transformers architectures, not the mixture of experts.)

| Name            | Hidden size | Dec. no. | Head no. | Max. length |
|-----------------|------------:|---------:|---------:|------------:|
| XGLM dense 125M |         768 |       12 |       12 |        2048 |
| XGLM dense 355M |        1024 |       24 |       16 |        2048 |
| XGLM dense 1.3B |        2048 |       24 |       32 |        2048 |
| XGLM dense 2.7B |        2560 |       32 |       32 |        2048 |
| XGLM dense 6.7B |        4096 |       32 |       32 |        2048 |
| XGLM dense 13B  |        5120 |       40 |       40 |        2048 |

### Feb 2022 - GPT-NeoX (EleutherAI) ###

GPT-NeoX is the third step towards a GPT-3 size open model by `EleutherAI`.
It was the largest open model at the time of its release, but far from the size of GPT-3 (175B).

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-NeoX-20B |        6144 |       44 |       64 |        2048 |       20,250 M |

### May 2022 - BLOOM ###

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) was created by over a thousand AI developers, organized by Hugging Face, published in May 2022.

| Name       | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------|------------:|---------:|---------:|------------:|---------------:|
| BLOOM-560M |        1024 |       24 |       16 |        2048 |          559 M | 
| BLOOM-1.1B |        1536 |       24 |       16 |        2048 |        1,065 M |
| BLOOM-1.7B |        2048 |       24 |       16 |        2048 |        1,722 M |
| BLOOM-3B   |        2560 |       30 |       32 |        2048 |        3,003 M |
| BLOOM-7.1B |        4096 |       30 |       32 |        2048 |        7,069 M |
| BLOOM-176B |       14336 |       70 |      112 |        2048 |      176,247 M |

### May 2022 - Meta (Facebook) OPT ###

Meta trained 9 models and made it accessible for research (non-commercial use) on 3 May 2022. (The 66B on 23 June.)
The largest model has an equivalent size to GPT-3. (It was the largest available model at the time.)

| Name     | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|----------|------------:|---------:|---------:|------------:|---------------:|
| OPT 125M |         768 |       12 |       12 |        2048 |          125 M | 
| OPT 350M |        1024 |       24 |       16 |        2048 |          331 M | 
| OPT 1.3B |        2048 |       24 |       32 |        2048 |        1,316 M |
| OPT 2.7B |        2560 |       32 |       32 |        2048 |        2,652 M |
| OPT 6.7B |        4096 |       32 |       32 |        2048 |        6,658 M |
| OPT 13B  |        5120 |       40 |       40 |        2048 |       12,853 M |
| OPT 30B  |        7168 |       48 |       56 |        2048 |       29,975 M |
| OPT 66B  |        9216 |       64 |       72 |        2048 |              ? |
| OPT 175B |       12288 |       96 |       96 |        2048 |              ? |

### Feb 2023 - LLaMA (Meta AI) ###

LLaMA (Large Language Model Meta AI) is a large language model announced by Meta (Facebook) on 23 Feb 2023. The trained parameters were shared only to academic researchers, but on 2 March it was leaked to the public.

Release dates: 
- Llama 1: 23 Feb 2023
- Llama 2: 18 July 2023
- Llama 3: 18 Apr 2024
- Llama 3.1: 23 July 2024
- Llama 3.2: 25 Sept 2024

All Llama 2 and 3 models has a base version, and a fine-tuned version for chat (instruct). 

The weights and the code are completely public for Llama 2 and 3, free to use even in commercial products. (Companies over 700 million monthly users has to request the licence.) 

Vocabulary size is 32,000 for Llama 1, 2; and 128,000 for Llama 3. 

| Name      | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|------------:|---------------:|
| LLaMA 7B  |        4096 |       32 |       32 |          32 |        2048 |        6,583 M |
| LLaMA 13B |        5120 |       40 |       40 |          40 |        2048 |       12,759 M |
| LLaMA 33B |        6656 |       60 |       52 |          52 |        2048 |       32,129 M |
| LLaMA 65B |        8192 |       80 |       64 |          64 |        2048 |       64,711 M |

Llama 2 uses an additional gate projection network in the MLP block, and the 70B model has Grouped Query Attention (GQA).

| Name        | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length |
|-------------|------------:|---------:|---------:|------------:|------------:|
| Llama 2 7B  |        4096 |       32 |       32 |          32 |        4096 |
| Llama 2 13B |        5120 |       40 |       40 |          40 |        4096 |
| Llama 2 70B |        8192 |       80 |       64 |           8 |        4096 |

Llama 3 architecturally the same as Llama 2-70B, all models use GQA (even the smallest).

| Name        | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length |
|-------------|------------:|---------:|---------:|------------:|------------:|
| Llama 3 8B  |        4096 |       32 |       32 |           8 |        8192 |
| Llama 3 70B |        8192 |       80 |       64 |           8 |        8192 |

Llama 3.1 is architecturally the same as Llama 3. The 405B is the largest open model. (Largest by far among the standard (non mixture-of-experts) models.)

| Name           | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length |
|----------------|------------:|---------:|---------:|------------:|------------:|
| Llama 3.1 8B   |        4096 |       32 |       32 |           8 |     128,000 |
| Llama 3.1 70B  |        8192 |       80 |       64 |           8 |     128,000 |
| Llama 3.1 405B |       16384 |      126 |      128 |          16 |     128,000 |

Llama 3.2 has text only and multimodal (image + text) variants as well. The latter is an encoder-decoder architecture, so that is different to that what is implemented here. The text-only models are the smallest Llama variants, architecturally the same as the 3.1 models. Text-only models:

| Name           | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length |
|----------------|------------:|---------:|---------:|------------:|------------:|
| Llama 3.2 1B   |        2048 |       16 |       32 |           8 |     128,000 |
| Llama 3.2 3B   |        3072 |       28 |       24 |           8 |     128,000 |

### March 2023 - GPT-4 (OpenAI) ###

GPT-4 was released in 14 March 2023, but almost all technical details are kept secret. It is known this is a Transformer architecture, using mixture of experts, and as input it can accept images as well. (The output is purely text.)

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|---------------:|
| GPT-4-8k  |          ?? |       ?? |       ?? |        8096 |           ?? M |
| GPT-4-32k |          ?? |       ?? |       ?? |       32768 |           ?? M |

### March 2023 - Cerebras ###

`Cerebras` released seven models in March 2023, trained on `Pile`. (https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models)

These are very similar models to the GPT-2/GTP-3 series, using the same tokenizer, same learned position embedding, etc. Unlike GPT-3, using always global attention.

| Name          | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|---------------|------------:|---------:|---------:|------------:|---------------:|
| Cerebras-111M |         768 |       10 |       12 |        2048 |          111 M |
| Cerebras-256M |        1088 |       14 |       17 |        2048 |          256 M |
| Cerebras-590M |        1536 |       18 |       12 |        2048 |          590 M |
| Cerebras-1.3B |        2048 |       24 |       16 |        2048 |         1316 M |
| Cerebras-2.7B |        2560 |       32 |       32 |        2048 |         2652 M |
| Cerebras-6.7B |        4096 |       32 |       32 |        2048 |         6658 M |
| Cerebras-13B  |        5120 |       40 |       40 |        2048 |        12853 M |

### Sep 2023 - Mistral ###

Mistral was released on 27 Sep 2023 by MistralAI. Later they created an instruct variant (fine-trained answering to instructions), and there are v.02 and v.03 versions as well. (At the newer versions the sparse attention was dropped.)

| Name            | Hidden size | Dec. no. | Head no. | Kv Head no. |Max. length |
|-----------------|------------:|---------:|---------:|------------:|-----------:|
| Mistral 7B v0.1 |        4096 |       32 |       32 |           1 |      32768 |

### Feb 2024 - Google Gemma ###

Gemma was announced and released on 21 Feb 2024. Gemma-2 was released on 27 June 2024.

| Name          | Hidden size | Dec. no. | Head no. | Kv Head no. | Max. length |
|---------------|------------:|---------:|---------:|------------:|------------:|
| Gemma 2B      |        2048 |       18 |        8 |           1 |        8192 |
| Gemma 7B      |        3072 |       28 |       16 |          16 |        8192 |
| Gemma 2B v1.1 |        2048 |       18 |        8 |           1 |        8192 |
| Gemma 7B v1.1 |        3072 |       28 |       16 |          16 |        8192 |
| Gemma-2 2.6B  |        2304 |       26 |        8 |           4 |        8192 |
| Gemma-2 9B    |        3584 |       42 |       16 |           8 |        8192 |
| Gemma-2 27B   |        4608 |       46 |       32 |          16 |        8192 |

### Grok ###

These are mixture of experts models, so this app cannot support these.