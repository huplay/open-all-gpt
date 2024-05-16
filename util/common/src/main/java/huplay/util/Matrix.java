package huplay.util;

public class Matrix
{
    private FloatType floatType;
    private Vector[] values;

    public Matrix(FloatType floatType, Vector[] values)
    {
        if (floatType.equals(FloatType.FLOAT32))
        {
            this.values = values;
        }
        if (floatType.equals(FloatType.FLOAT16))
        {

        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    public Vector[] getValues()
    {
        return values;
    }

    public Vector get(int index)
    {
        return values[index];
    }
}
