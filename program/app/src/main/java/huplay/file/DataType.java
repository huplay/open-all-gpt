package huplay.file;

public enum DataType
{
    BOOLEAN(1),

    SIGNED_BYTE(8),
    UNSIGNED_BYTE(8),

    SIGNED_INTEGER_16(16),
    UNSIGNED_INTEGER_16(16),

    SIGNED_INTEGER_32(32),
    UNSIGNED_INTEGER_32(32),

    SIGNED_INTEGER_64(64),
    UNSIGNED_INTEGER_64(64),

    FLOAT_16(16),
    BRAIN_FLOAT_16(16),

    FLOAT_32(32),

    FLOAT_64(64);

    private final int bits;

    DataType(int bits)
    {
        this.bits = bits;
    }

    // Getters
    public int getBits() {return bits;}
}
