package math.dataType;

/**
 * 32-bit float constants
 * Structure: 1 sign bit + 8 exponent bits + 23 fraction bits
 */
public class Float32
{
    public static final int EXPONENT_BITS = 8;
    public static final int MANTISSA_BITS = 23;

    public static final int EXPONENT_OFFSET = 127;

    public static final float POSITIVE_INFINITY = Float.intBitsToFloat(0b0_11111111_00000000000000000000000);
    public static final float NEGATIVE_INFINITY = Float.intBitsToFloat(0b1_11111111_00000000000000000000000);
    public static final float NaN               = Float.intBitsToFloat(0b0_11111111_10000000000000000000000);
}
