package huplay.dataType.matrix;

public enum MatrixType
{
    /**********************************************************
     * Standard matrix types which be used for internal storage
     * (in matrices created by the transformer logic):
     **********************************************************/
    VECTOR_ARRAY_FLOAT_32, // Vector array, Float 32 values
    VECTOR_ARRAY_FLOAT_16, // Vector array, Float 16 values
    VECTOR_ARRAY_BRAIN_FLOAT_16, // Vector array, Brain Float 16 values

    /**************************************************************************
     * Quantized matrix types, can be used only storing the trained parameters:
     **************************************************************************/

    // LLM.int8() quantization (University of Washington, Facebook AI Research, Hugging Face, ENS Paris-Saclay)
    // Publication (15 Aug 2022): https://arxiv.org/abs/2208.07339
    LLM_INT_8,

    // GPTQ (GPT Post-Training Quantization) (ETH Zurich, IST Austria, NeuralMagic)
    // Publication (31 Oct 2022): https://arxiv.org/abs/2210.17323
    // Code for the publication: https://github.com/IST-DASLab/gptq
    GPTQ_4, // GPTQ 4 bit
    GPTQ_3, // GPTQ 3 bit
    GPTQ_2, // GPTQ 2 bit

    // QLoRA (Quantized Low Rank Adapters) (University of Washington)
    // Publication (23 May 2023): https://arxiv.org/abs/2305.14314
    // Huggingface blog post: https://huggingface.co/blog/4bit-transformers-bitsandbytes
    QLoRA_INT_8, // QLoRA Int 8 bit
    QLoRA_PF4, // QLoRA pure float 4 bit
    QLoRA_NF4, // QLoRA normal float 4 bit
    QLoRA_NF4_DQ, // QLoRA normal float 4 bit with double quantization

    AWQ,

    // Half-Quadratic Quantization
    HQQ_2,
    HQQ_1,
    HQQ_PLUS_2,
    HQQ_PLUS_1,

    QUIP_4,
    QUIP_2,
    QUIP_HASH_4,
    QUIP_HASH_2,

    // https://towardsdatascience.com/the-aqlm-quantization-algorithm-explained-8cf33e4a783e
    AQLM_3, // Additive Quantization of Language Models
    ALQM_2,

    SpQR_3
}
