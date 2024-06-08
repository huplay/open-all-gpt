package huplay;

import huplay.math.AbstractMathUtility;

public class MathProvider
{
    private static AbstractMathUtility MATH_UTILITY;

    public static void setMathUtility(AbstractMathUtility mathUtility)
    {
        MathProvider.MATH_UTILITY = mathUtility;
    }

    public static AbstractMathUtility getMathUtility()
    {
        return MATH_UTILITY;
    }
}
