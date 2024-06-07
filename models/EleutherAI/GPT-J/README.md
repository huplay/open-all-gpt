# History of the Rotary Position Embedding

EleutherAI GPT-J 6B was the first large model uses the Rotary Position Embedding (RoPE). It was implemented using Ben Wang's Mesh Transformer JAX, which adopted the RoFormer position embedding, created by the ZhuiyiTechnology (Shenzhen, China). 

Original implementation:
- Code (ZhuiyiTechnology, first commit in 22 Mar 2021): https://github.com/ZhuiyiTechnology/roformer
- Blog post (By Su Jianlin, ZhuiyiTechnology, in Chinese, 23 Mar 2021): https://kexue.fm/archives/8265
- Official publication (RoFormer) (in English, 20 Apr 2021, Jianlin Su et al., ZhuiyiTechnology): https://arxiv.org/abs/2104.09864

The above positional embedding was added to Mesh Transformer JAX (By Ben Wang)
- Repo: https://github.com/kingoflolz/mesh-transformer-jax/
- Commit ("add rotary pos encoding, make pos encoding configurable") in 18 Apr, 2021: https://github.com/kingoflolz/mesh-transformer-jax/commit/728d20b26785495852c8851ce0d770bc10d3caf0

GPT-J was released in June 2021. (Using Ben Wang's Mesh Transformer JAX.)

Later it was added to:
 - Pytorch (29 Jun - 16 Aug 2021): https://github.com/lucidrains/rotary-embedding-torch
 - Transformers of Hugging Face (31 Aug 2021): https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gptj/modeling_gptj.py
 - Hugging Face documentation (RoFormer): https://huggingface.co/docs/transformers/model_doc/roformer

Later it was used in EleutherAI GPT-NEOX, Meta Llama and Google Gemini/Gemma as well.





