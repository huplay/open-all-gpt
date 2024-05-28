package huplay.parameters.safetensors;

import huplay.dataType.DataType;

public enum SafetensorsDataType
{
    BOOL(1, DataType.BOOLEAN),

    I8(8, DataType.BYTE),
    U8(8, DataType.UNSIGNED_BYTE),

    I16(16, DataType.INTEGER_16),
    U16(16, DataType.UNSIGNED_INTEGER_16),

    I32(32, DataType.INTEGER_32),
    U32(32, DataType.UNSIGNED_INTEGER_32),

    I64(64, DataType.INTEGER_64),
    U64(64, DataType.UNSIGNED_INTEGER_64),

    F16(16, DataType.FLOAT_16),
    BF16(16, DataType.BRAIN_FLOAT_16),

    F32(32, DataType.FLOAT_32),

    F64(64, DataType.FLOAT_64);

    private final int bits;
    private final DataType dataType;

    SafetensorsDataType(int bits, DataType dataType)
    {
        this.bits = bits;
        this.dataType = dataType;
    }

    // Getters
    public int getBits() {return bits;}
    public DataType getDataType() {return dataType;}
}
