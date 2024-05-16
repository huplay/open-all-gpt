# BLOOM #

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) was created by over a thousand AI developers, organized by Hugging Face, published in May 2022.

https://bigscience.huggingface.co/

Publication: https://arxiv.org/abs/2211.05100

| Name       | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------|------------:|---------:|---------:|------------:|---------------:|
| BLOOM-560M |        1024 |       24 |       16 |        2048 |          559 M | 
| BLOOM-1.1B |        1536 |       24 |       16 |        2048 |        1,065 M |
| BLOOM-1.7B |        2048 |       24 |       16 |        2048 |        1,722 M |
| BLOOM-3B   |        2560 |       30 |       32 |        2048 |        3,003 M |
| BLOOM-7.1B |        4096 |       30 |       32 |        2048 |        7,069 M |
| BLOOM-176B |       14336 |       70 |      112 |        2048 |      176,247 M |

(The BLOOM-176B model is slightly larger than the largest GPT-3, no way you will be able to use it with this application.)

There was an attempt to train a BLOOM-104B version, but it was a failure because of numerical instabilities.
Too large numbers were calculated for the FLOAT16 data type, which has only a 5 bits long exponent. (10 bits mantissa.)
The additional input embedding was an attempt to solve this issue, which worked for the smaller models, but not for the BLOOM-104B.
Finally, they opted to use the BFLOAT16 data type, which is also 16-bit format, but the exponent is larger, 8 bits long. (The mantissa is reduced to 7 bits)
The BLOOM-176B model was trained using this data type, while the potentially unnecessary input normalization remained in use. 

The models were uploaded to the `Hugging Face` portal where you can find the links for all models: https://huggingface.co/docs/transformers/model_doc/bloom

## BLOOMZ ##

There is a series of BLOOM, which is fine-tuned for instructions. These are the same models as the original BLOOM, the change is only in the parameters.
So it is possible to use them in the app as well:

- https://huggingface.co/bigscience/bloomz-1b1
- https://huggingface.co/bigscience/bloomz-1b7
- https://huggingface.co/bigscience/bloomz-560m
- https://huggingface.co/bigscience/bloomz-3b