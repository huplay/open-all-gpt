package dataType;

import math.dataType.Float32;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class Float32Test
{
    @Test
    public void testConstants()
    {
        assertEquals(Float.POSITIVE_INFINITY, Float32.POSITIVE_INFINITY, 0);
        assertEquals(Float.NEGATIVE_INFINITY, Float32.NEGATIVE_INFINITY, 0);
        assertEquals(Float.NaN, Float32.NaN, 0);
    }

    @Test
    public void testExample()
    {
        // How the 0.375 (decimal) is stored?
        // https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Converting_decimal_to_binary32

        // In binary this is: 0.011, because this is 0 * 1/2 + 1 * 1/4 + 1 * 1/8 = 0 + 0.25 + 0.125 = 0.375
        // Shifting the fraction point to the right by two steps gives the normalized format: 1.1 * 2e-2
        // (The 1.1 is in binary, the 2e-2 is written in decimal.)

        // In float (FLOAT_32) it is stored by the following bits:    00111110 11000000 00000000 00000000
        // Splitting by the float 32 structure:                       0 01111101 10000000000000000000000

        // Because:
        // Sign bit: 0 (-> positive)
        // Exponent: 01111101 = 125 (extracting 127 -> -2) (this gives the "e-2" part)
        // Fraction: 10000000000000000000000 -> 1.1 (The leading 1 is added automatically if the exponent != -127)
        // (The fraction digits are the digits after the fraction point, so the trailing zeroes can be omitted.)

        // Test:
        int example = 0b00111110_11000000_00000000_00000000;
        float floatValue = Float.intBitsToFloat(example); // Int bits are converted to float

        assertEquals(0.375f, floatValue, 0);
    }
}
