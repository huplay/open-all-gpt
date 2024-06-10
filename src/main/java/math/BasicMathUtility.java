package math;

public class BasicMathUtility
{
    // Basic math functions to return float instead of double
    public static float pow(float a, float b) {return (float)(java.lang.Math.pow(a, b));}
    public static float exp(double value) {return (float)(java.lang.Math.exp(value));}
    public static float sqrt(float value) {return (float)(java.lang.Math.sqrt(value));}
    public static float cos(double value) {return (float)(java.lang.Math.cos(value));}
    public static float sin(double value) {return (float)(java.lang.Math.sin(value));}

    public static float absMax(float[] values)
    {
        var max = 0f;
        for (var value : values)
        {
            if (Math.abs(value) > max)
            {
                max = Math.abs(value);
            }
        }

        return max;
    }
}
