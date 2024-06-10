package dataType;

import math.dataType.Float16;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class Float16Test
{
    @Test
    public void testExample()
    {
        // How the 0.33325195 (decimal) is stored? (Nearest value to 1/3)
        // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples

        // In binary this is: 0.0101010101
        // Shifting the fraction point to the right by two steps gives the normalized format: 1.01010101 * 2e-2
        // (The 1.01010101 is in binary, the 2e-2 is written in decimal.)

        // In 16-bit float (FLOAT_16) it is stored by the following bits:   00110101 01010101
        // Or splitting differently, matching to the FLOAT 16 structure:    0 01101 0101010101

        // Because:
        // Sign bit: 0 (-> positive)
        // Exponent: 01101 = 13 (extracting 15 -> -2) (this gives the "e-2" part)
        // Fraction: 0101010101 -> 1.0101010101 (The leading 1 is added automatically if the exponent != -127)
        // (The fraction digits are the digits after the fraction point, so the trailing zeroes can be omitted.)

        // Test toFloat32:
        short shortValue = (short) 0b00110101_01010101; // = 0x3555
        float floatValue = Float16.toFloat32(shortValue);
        AssertFloat.assertFloat(0.33325195f, floatValue);

        float floatValue2 = Float16.toFloat32Readable(shortValue);
        AssertFloat.assertFloat(0.33325195f, floatValue);

        // Test toShort:
        short value = Float16.toShort(floatValue);
        assertEquals(0x3555, value);

        short value2 = Float16.toShortReadable(floatValue);
        assertEquals(0x3555, value2);
    }

    @Test
    public void testConversions()
    {
        var values = new float[] {0, 100, -100, 1.234f, -9.7655f, 1.5e4f, -1.5e3f, 5e-4f,
                                  Float.NEGATIVE_INFINITY, Float.NaN};

        for (var value : values)
        {
            short shortValue = Float16.toShort(value);
            float floatValue = Float16.toFloat32(shortValue);
            AssertFloat.assertFloat(value, floatValue);

            short shortValue2 = Float16.toShortReadable(value);
            float floatValue2 = Float16.toFloat32Readable(shortValue2);
            AssertFloat.assertFloat(value, floatValue2);
        }
    }
}
