package huplay.dataType;

/**
 * Google Brain Float 16 (BFLOAT16) floating-point data type implementation
 * (It isn't supported in Java, so it is stored in a short value)
 * 1 sign bit + 8 exponent bits + 7 fraction bits
 */
public class BrainFloat16
{
    public static final short POSITIVE_INFINITY = (short) 0b0_11111111_0000000;
    public static final short NEGATIVE_INFINITY = (short) 0b1_11111111_0000000;
    public static final short NaN               = (short) 0b0_11111111_0000001;

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
     * Conversion from 32-bit float:        1 sign bit + 8 exponent bits + 23 fraction bits
     *            to 16-bit brain float:    1 sign bit + 8 exponent bits + 7 fraction bits

     * The sign bit and the exponent part will be the same, just the fraction will be longer:
     * The first 16 bits of the fraction will be filled with zeroes.
     */
    public static short toShort(float value)
    {
        int intBits = Float.floatToRawIntBits(value);

        //int signFlag = intValue & 0b1000_0000_0000_0000_0000_0000_0000_0000; // Extract the sign bit (1st bit)
        //int exponent = intValue & 0b0111_1111_1000_0000_0000_0000_0000_0000; // Extract the exponent (8 bits after sign)
        //int mantissa = intValue & 0b0000_0000_0111_1111_1111_1111_1111_1111; // Extract the mantissa (last 23 bits)

        var signFlag = (intBits >>> 16) & 0x8000;
        var exponent = (intBits >>> 16) & 0x7f80;
        var mantissa = (intBits & 0x7fffff);

        if (exponent == 0x7f80)
        {
            if (mantissa != 0)
            {
                return NaN;
            }
            else
            {
                return (short) (intBits >>> 16);
            }
        }
        else
        {
            var m1 = round(mantissa, 16);
            var e1 = exponent + m1;
            return (short) (signFlag | e1);
        }
    }

    private static int round(int value, int shifts)
    {
        var mid = 1 << (shifts - 1);
        var mask = (1 << shifts) - 1;
        var mshift = value >> shifts;
        var masked = value & mask;
        var cmp = masked - mid;

        // we are losing more than 1/2
        if (cmp > 0)
        {
            return mshift + 1;
        }
        else if (cmp < 0)
        {
            // we are losing < 1/2
            return mshift;
        }
        else
        {
            // we are losing exactly 1/2
            // we round to the nearest even
            // 2.5 => 2, 3.5 => 4, 4.5 => 4
            // -2.5 => -2, -3.5 => -4, -4.5 => -4
            var isOdd = (mshift & 1) != 0;
            if (isOdd)
            {
                return mshift + 1;
            }
            else
            {
                return mshift;
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
