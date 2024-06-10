package dataType;

import math.dataType.BrainFloat16;
import org.junit.Test;

import static dataType.AssertFloat.assertFloat;
import static org.junit.Assert.assertEquals;

public class BrainFloat16Test
{
    @Test
    public void testExample()
    {
        // How the 3.140625 (decimal) is stored? (Nearest value to PI)
        // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples

        // In binary this is: 11.001001 (1*2 + 1*1 + 1*1/8 + 1*1/64 = 3 + 0.125 + 0.015625
        // Shifting the fraction point to the left by one step gives the normalized format: 1.1001001 * 2e1
        // (The 1.1001001 is in binary, the 2e1 (=2) is written in decimal.)

        // In 16-bit brain float (BFLOAT_16) it is stored by the following bits:   01000000 01001001
        // Or splitting differently, matching to the FLOAT 16 structure:           0 10000000 1001001

        // Because:
        // Sign bit: 0 (-> positive)
        // Exponent: 10000000 = 128 (extracting 127 -> 1) (this gives the "e1" part)
        // Fraction: 1001001 -> 1.1001001 (The leading 1 is added automatically if the exponent != -127)
        // (The fraction digits are the digits after the fraction point, so the trailing zeroes can be omitted.)

        // Test toFloat32:
        short shortValue = (short) 0b01000000_01001001; // = 0x4049
        float floatValue = BrainFloat16.toFloat32(shortValue);
        assertFloat(3.140625f, floatValue);

        // Test toShort:
        short value = BrainFloat16.toShort(floatValue);
        assertEquals(0x4049, value, 0);
    }

    @Test
    public void testConversions()
    {
        var values = new float[] {0, 100, -100, 1.234f, -9.7655f, 1.5e13f, -1.5e3f, 5e-4f,
                                  Float.NEGATIVE_INFINITY, Float.NaN};

        for (var value : values)
        {
            short shortValue = BrainFloat16.toShort(value);
            float floatValue = BrainFloat16.toFloat32(shortValue);
            assertFloat(value, floatValue);
        }
    }
}
