package math.dataType;

/**
 * Float 16 floating-point data type implementation (IEEE 754-2008 Half-precision)
 * (It isn't supported in Java, so it is stored in a short value)
 * 1 sign bit + 5 exponent bits + 10 fraction bits
 */
public class Float16
{
    public static final int EXPONENT_BITS = 5;
    public static final int MANTISSA_BITS = 10;

    public static final int EXPONENT_OFFSET = 15;

    public static final short POSITIVE_INFINITY = (short) Float.intBitsToFloat(0b0_11111_0000000000);
    public static final short NEGATIVE_INFINITY = (short) Float.intBitsToFloat(0b1_11111_0000000000);
    public static final short NaN               = (short) Float.intBitsToFloat(0b0_11111_1000000000);

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
        // The 1st bit is the sign. If it is zero, the number is positive.
        boolean isPositive = (value & 0b1000_0000_0000_0000) == 0;

        // Extract the exponent (5 bits after the sign bit), and remove the EXPONENT_OFFSET
        int exponent = ( (value & 0b0111_1100_0000_0000) >> 10 ) - EXPONENT_OFFSET;

        // Extract the mantissa (last 10 bits)
        int mantissa = value & 0b0000_0011_1111_1111;

        if (exponent == 16)
        {
            if (mantissa == 0)
            {
                // Infinity
                return isPositive ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            else
            {
                // Not a Number (NaN)
                // Every ?111_11??_????_???? is a NaN (except ?111_1100_0000_0000), so there are 2047 NaN values
                return Float.NaN; // We will convert all of these to a single NaN value (the NaN payload is lost)
            }
        }
        else if (exponent == -15)
        {
            // Zero or subnormal values
            return (isPositive ? 1 : -1) * 0x1p-24f * mantissa;

            /*if (mantissa != 0)
            {
                exponent = 0x1c400; // 0001_1100_0100_0000_0000
                do
                {
                    mantissa <<= 1; // Shift left the mantissa (same as multiplied by 2)
                    exponent -= 0b0000_0100_0000_0000; // 1024
                }
                while ((mantissa & 0b0000_0100_0000_0000) == 0);

                mantissa &= 0b0000_0011_1111_1111;
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);*/
        }
        else
        {
            // Normal values

            // Set the first bit to 1 if negative
            int float32Sign = isPositive ? 0 : (1 << 31);

            // Add the Float32 exponent offset to the exponent and shift it to the right place (before the mantissa)
            int float32Exponent = (exponent + Float32.EXPONENT_OFFSET) << Float32.MANTISSA_BITS;

            // Shift the mantissa to the beginning of the mantissa bits (the remaining values will be zero)
            int float32Mantissa = mantissa << (Float32.MANTISSA_BITS - MANTISSA_BITS);

            // Build the float 32 value from the bits
            return Float.intBitsToFloat(float32Sign | float32Exponent | float32Mantissa);
        }
    }
}
