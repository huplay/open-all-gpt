# GPT-NEO #

`EleutherAI` is attempted to recreate all the GPT-3 variants, training them on their own dataset (`Pile`). (https://www.eleuther.ai)

They started the work on the smallest models, which are similar in size to the GPT-2.
I called the GPT-2 models as SMALL/MEDIUM/LARGE/XL. They officially released a model similar to the SMALL (NEO-125M) and XL (NEO-1.3B).
They trained a MEDIUM model as well (NEO-350M), which is mentioned occasionally, but it isn't uploaded to the official page. (I found only a copy uploaded by someone else.)
After these they've done a bigger model which is similar in size to the GPT-3 Ada (NEO-2.7B). (That is the smallest GPT-3 model that is available for the public.)

These above-mentioned models are under the NEO series.
(Later they modified their code base and some implementation details, so the larger models are under the GPT-J and GPT-NEOX series.)

| Name                               | Hidden size | Dec. no. | Head no. | Max. length | Size of params |                                                        |
|------------------------------------|------------:|---------:|---------:|------------:|---------------:|--------------------------------------------------------|
| GPT-NEO-SMALL <br /> GPT-NEO-125M  |         768 |       12 |       12 |        2048 |          124 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-125M) |
| GPT-NEO-MEDIUM <br /> GPT-NEO-350M |        1024 |       24 |       16 |        2048 |          355 M | [Link](https://huggingface.co/xhyi/PT_GPTNEO350_ATG)   |
| GPT-NEO-XL <br /> GPT-NEO-1.3B     |        2048 |       24 |       16 |        2048 |        1,314 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-1.3B) |
| GPT-NEO-ADA <br /> GPT-NEO-2.7B    |        2560 |       32 |       20 |        2048 |        2,649 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-2.7B) |
