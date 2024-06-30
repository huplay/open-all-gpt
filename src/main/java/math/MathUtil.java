package math;

import app.IdentifiedException;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public enum MathUtil
{
    STANDARD("math.impl.standard.StandardMath"),
    ND4J("math.impl.nd4j.ND4JMath"),
    VECTOR_API("math.impl.vectorApi.VectorApiMath");

    public static final AbstractMathUtility MATH = MathUtil.getInstance();

    private final String className;

    MathUtil(String className)
    {
        this.className = className;
    }

    public String getClassName()
    {
        return className;
    }

    public static AbstractMathUtility getInstance()
    {
        try (var in = MathUtil.class.getResourceAsStream("/math.provider"))
        {
            var reader = new BufferedReader(new InputStreamReader(in));
            var provider = reader.readLine().toUpperCase();
            Class<?> providerClass = Class.forName(MathUtil.valueOf(provider).getClassName());

            var instance = providerClass.getDeclaredConstructor().newInstance();

            if (instance instanceof AbstractMathUtility mathUtility)
            {
                return mathUtility;
            }
            else
            {
                throw new IdentifiedException("Incorrect math utility");
            }
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Error during instantiating requested math provider", e);
        }
    }
}
