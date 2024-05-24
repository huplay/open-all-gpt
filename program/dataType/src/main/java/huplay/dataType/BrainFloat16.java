package huplay.dataType;

/**
 * Google Brain Float 16 (BFLOAT16) floating-point data type implementation
 * (It isn't supported in Java, so it is stored in a short value)
 * 1 sign bit + 8 exponent bits + 7 fraction bits
 */
public class BrainFloat16
{
    public static final int EXPONENT_BITS = 8;
    public static final int MANTISSA_BITS = 7;

    public static final int EXPONENT_OFFSET = 127;

    public static final short POSITIVE_INFINITY = (short) Float.intBitsToFloat(0b0_11111111_0000000);
    public static final short NEGATIVE_INFINITY = (short) Float.intBitsToFloat(0b1_11111111_0000000);
    public static final short NaN               = (short) Float.intBitsToFloat(0b0_11111111_1000000);

    private final short value;

    public BrainFloat16(short value)
    {
        this.value = value;
    }

    public BrainFloat16(float value)
    {
        this.value = toShort(value);
    }

    public short getValue()
    {
        return value;
    }

    public float getFloatValue()
    {
        return toFloat32(value);
    }

    /**
     * Conversion from 32-bit float:        1 sign bit + 8 exponent bits + 23 mantissa bits
     *            to 16-bit brain float:    1 sign bit + 8 exponent bits + 7 mantissa bits

     * The sign bit and the exponent part will be the same, the mantissa should be rounded from 23 bits to 7
     */
    public static short toShort(float value)
    {
        int intBits = Float.floatToRawIntBits(value);

        int signFlag = (intBits >>> 16) & 0b1_00000000_0000000;
        int exponent = (intBits >>> 16) & 0b0_11111111_0000000;
        int mantissa = (intBits & 0b0_00000000_11111111111111111111111);

        if (exponent == 0b0_11111111_0000000)
        {
            if (mantissa != 0)
            {
                return (short) 0b0_11111111_1000000;
            }
            else
            {
                return (short) (intBits >>> 16);
            }
        }
        else
        {
            var roundedMantissa = roundMantissa(mantissa);
            return (short) (signFlag | (exponent + roundedMantissa));
        }
    }

    private static int roundMantissa(int value)
    {
        var shiftedValue = value >> 16;
        var cmp = (value & 0b1111_1111_1111_1111) - 0b1000_0000_0000_0000;

        // we are losing more than 1/2
        if (cmp > 0)
        {
            return shiftedValue + 1;
        }
        else if (cmp < 0)
        {
            // we are losing < 1/2
            return shiftedValue;
        }
        else
        {
            // we are losing exactly 1/2
            // we round to the nearest even
            // 2.5 => 2, 3.5 => 4, 4.5 => 4
            // -2.5 => -2, -3.5 => -4, -4.5 => -4
            if ((shiftedValue & 1) != 0)
            {
                return shiftedValue + 1;
            }
            else
            {
                return shiftedValue;
            }
        }
    }

    /**
     * Conversion from 16-bit brain float to 32-bit float
     */
    public static float toFloat32(short value)
    {
        // Because these have the same exponent size, the conversion is simple:
        // All bits should remain the same, appended by 16 zeroes.
        return Float.intBitsToFloat(value << 16);
    }
}
