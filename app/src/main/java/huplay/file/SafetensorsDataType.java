package huplay.file;

public enum SafetensorsDataType
{
    // Boolean
    BOOL(1),

    // Unsigned byte
    U8(8),

    // Signed byte
    I8(8),

    // 16-bit signed integer
    I16(16),

    // 16-bit unsigned integer
    U16(16),

    // 16-bit float
    F16(16),

    // 16-bit brain float
    BF16(16),

    // 32-bit signed integer
    I32(32),

    // 32-bit unsigned integer
    U32(32),

    // 32-bit float
    F32(32),

    // 64-bit float
    F64(64),

    // 64-bit signed integer
    I64(64),

    // 64-bit unsigned integer
    U64(64);

    private final int bits;

    SafetensorsDataType(int bits)
    {
        this.bits = bits;
    }

    public int getBits()
    {
        return bits;
    }
}
