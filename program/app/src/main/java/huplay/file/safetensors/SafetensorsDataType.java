package huplay.file.safetensors;

import huplay.file.DataType;

public enum SafetensorsDataType
{
    BOOL(1, DataType.BOOLEAN),

    I8(8, DataType.SIGNED_BYTE),
    U8(8, DataType.UNSIGNED_BYTE),

    I16(16, DataType.SIGNED_INTEGER_16),
    U16(16, DataType.UNSIGNED_INTEGER_16),

    I32(32, DataType.SIGNED_INTEGER_32),
    U32(32, DataType.UNSIGNED_INTEGER_32),

    I64(64, DataType.SIGNED_INTEGER_64),
    U64(64, DataType.UNSIGNED_INTEGER_64),

    F16(16, DataType.FLOAT_16),
    BF16(16, DataType.BRAIN_FLOAT_16),

    F32(32, DataType.FLOAT_32),

    F64(64, DataType.FLOAT_64);

    private final int bits;
    private DataType dataType;

    SafetensorsDataType(int bits, DataType dataType)
    {
        this.bits = bits;
        this.dataType = dataType;
    }

    // Getters
    public int getBits() {return bits;}
    public DataType getDataType() {return dataType;}
}
