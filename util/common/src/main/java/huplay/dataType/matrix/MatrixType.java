package huplay.dataType.matrix;

public enum MatrixType
{
    VECTOR_ARRAY_FLOAT_32,
    VECTOR_ARRAY_FLOAT_16,
    VECTOR_ARRAY_BRAIN_FLOAT_16,

    // https://arxiv.org/pdf/2305.14314
    // https://huggingface.co/blog/4bit-transformers-bitsandbytes
    QLoRA_NORMAL_FLOAT_4,
    QLoRA_NORMAL_FLOAT_4_DOUBLE
}
