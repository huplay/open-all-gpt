package huplay;

import huplay.math.MathUtility;

public class MathUtilProvider
{
    public static final MathUtility MATH = new MathUtility();

    static
    {
        MathProvider.setMathUtility(MATH);
    }
}
