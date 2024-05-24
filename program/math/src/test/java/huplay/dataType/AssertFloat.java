package huplay.dataType;

import static org.junit.Assert.assertEquals;

public class AssertFloat
{
    public static void assertFloat(float expected, float actual)
    {
        if (Float.isNaN(expected))
        {
            assertEquals(Float.NaN, actual, 0);
        }
        else if (expected == Float.POSITIVE_INFINITY)
        {
            assertEquals(Float.POSITIVE_INFINITY, actual, 0);
        }
        else if (expected == Float.NEGATIVE_INFINITY)
        {
            assertEquals(Float.NEGATIVE_INFINITY, actual, 0);
        }
        else
        {
            assertEquals(Math.signum(expected), Math.signum(actual), 0);
            assertEquals(Math.getExponent(expected), Math.getExponent(actual));
            assertEquals(getMantissa(expected), getMantissa(actual), 0.004f);
        }
    }

    public static double getMantissa(float value)
    {
        int exponent = Math.getExponent(value);
        return value / Math.pow(2, exponent);
    }
}
