package huplay.dataType;

/**
 * Float 16 floating-point data type implementation (IEEE 754-2008 Half-precision)
 * (It isn't supported in Java, so it is stored in a short value)
 * 1 sign bit + 5 exponent bits + 10 fraction bits
 */
public class Float16
{
    public static final short POSITIVE_INFINITY = (short) 0b0_11111_0000000000;
    public static final short NEGATIVE_INFINITY = (short) 0b1_11111_0000000000;

    private final short value;

    public Float16(short value)
    {
        this.value = value;
    }

    public Float16(float value)
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
     * Conversion from 32-bit float to 16-bit float (stored in a short value)
     */
    public static short toShort(float value)
    {
        return Float.floatToFloat16(value);
    }

    /**
     * Conversion from 16-bit float (stored in a short value) to 32-bit float
     */
    public static float toFloat32(short value)
    {
        return Float.float16ToFloat(value);
    }

    /**
     * Conversion from 32-bit float to 16-bit float (stored in a short value)
     * (This is the same as toShort(), but in a more readable implementation.)
     */
    public static short toShortReadable(float value)
    {
        int floatBits = Float.floatToRawIntBits(value);
        int signFlag = floatBits >>> 16 & 0x8000;
        int val = (floatBits & 0x7fffffff) + 0x1000; // rounded value

        if (val >= 0x47800000) // 0100_0111_1000_0000_0000_0000_0000_0000
        {
            // Infinity or NaN
            if ( (floatBits & 0x7fffffff) >= 0x47800000)
            {                                 // is or must become NaN/Inf
                if (val < 0x7f800000) // 0111_1111_1000_0000_0000_0000_0000_0000 was value but too large
                {
                    if (signFlag == 0)
                    {
                        return POSITIVE_INFINITY;
                    }
                    else
                    {
                        return NEGATIVE_INFINITY;
                    }
                }

                return (short) (signFlag | 0x7c00 | ( floatBits & 0x007fffff ) >>> 13); // remains +/-Inf or NaN | keep NaN (and Inf) bits
            }

            return (short) (signFlag | 0x7bff);             // unrounded not quite Inf
        }

        if (val >= 0x38800000)               // remains normalized value
        {
            return (short) (signFlag | val - 0x38000000 >>> 13); // exp - 127 + 15
        }

        if (val < 0x33000000)                // too small for subnormal
        {
            return (short) signFlag;                      // becomes +/-0
        }

        val = (floatBits & 0x7fffffff) >>> 23;  // tmp exp for subnormal calc

        return (short) (signFlag | ( (floatBits & 0x7fffff | 0x800000) // add subnormal bit
                + (0x800000 >>> val - 102)     // round depending on cut off
                >>> 126 - val ));   // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }

    /**
     * Conversion from 16-bit float (stored in a short value) to 32-bit float
     * (This is the same as toFloat32(), but in a more readable implementation.)
     */
    public static float toFloat32Readable(short value)
    {
        var signFlag = value & 0b1000_0000_0000_0000; // Extract the sign bit (1st bit)
        var exponent = value & 0b0111_1100_0000_0000; // Extract the exponent (5 bits after sign)
        var mantissa = value & 0b0000_0011_1111_1111; // Extract the mantissa (last 10 bits)

        if (exponent == 0b0111_1100_0000_0000)
        {
            // Infinity or NaN
            if (mantissa == 0)
            {
                if (signFlag == 0)
                {
                    return Float.POSITIVE_INFINITY;
                }
                else
                {
                    return Float.NEGATIVE_INFINITY;
                }
            }
            else return Float.NaN; // Interestingly, there are 2048 different FLOAT16 NaN values
        }
        else if (exponent == 0)
        {
            // Zero or subnormal values
            if (mantissa != 0)
            {
                exponent = 0x1c400; // 0001_1100_0100_0000_0000
                do
                {
                    mantissa = mantissa << 1;
                    exponent -= 0b0000_0100_0000_0000; // 1024
                }
                while ((mantissa & 0b0000_0100_0000_0000) == 0);

                mantissa &= 0b0000_0011_1111_1111;
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
        else
        {
            // Normal values
            exponent += 0x1c000;
            if (mantissa == 0 && exponent > 0x1c400)
            {
                return Float.intBitsToFloat(signFlag << 16 | exponent << 13 | 0b0000_0011_1111_1111);
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
    }
}
