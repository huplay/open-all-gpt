package huplay.util;

public class Vector
{
    private final FloatType floatType;
    private float[] float32Values;
    private short[] float16Values;

    public Vector(FloatType floatType, int size)
    {
        this.floatType = floatType;
        if (floatType.equals(FloatType.FLOAT32))
        {
            float32Values = new float[size];
        }
        else if (floatType.equals(FloatType.FLOAT16))
        {
            float16Values = new short[size];
        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
            float16Values = new short[size];
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    public Vector(FloatType floatType, float[] values)
    {
        this.floatType = floatType;
        if (floatType.equals(FloatType.FLOAT32))
        {
            float32Values = values;
        }
        else if (floatType.equals(FloatType.FLOAT16))
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

    public Vector(FloatType floatType, short[] values)
    {
        this.floatType = floatType;
        if (floatType.equals(FloatType.FLOAT32))
        {
        }
        else if (floatType.equals(FloatType.FLOAT16))
        {
            float16Values = values;
        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    public static Vector[] newVectorArray(FloatType floatType, int rows, int cols)
    {
        Vector[] matrix = new Vector[rows];

        for (int i = 0; i < rows; i++)
        {
            matrix[i] = new Vector(floatType, cols);
        }

        return matrix;
    }

    public FloatType getFloatType()
    {
        return floatType;
    }

    public float[] getFloat32Values()
    {
        return float32Values;
    }

    public short[] getFloat16Values()
    {
        return float16Values;
    }

    public float get(int index)
    {
        if (floatType.equals(FloatType.FLOAT32))
        {
            return float32Values[index];
        }
        else if (floatType.equals(FloatType.FLOAT16))
        {
            return 1; // TODO
        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
            return 1; // TODO
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    public void set(int index, float value)
    {
        if (floatType.equals(FloatType.FLOAT32))
        {
            float32Values[index] = value;
        }
        else if (floatType.equals(FloatType.FLOAT16))
        {
            float16Values[index] = 1; // TODO
        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
            float16Values[index] = 1; // TODO
        }
    }

    public int size()
    {
        if (floatType.equals(FloatType.FLOAT32))
        {
            return float32Values.length;
        }
        else if (floatType.equals(FloatType.FLOAT16))
        {
            return float16Values.length;
        }
        else if (floatType.equals(FloatType.BFLOAT16))
        {
            return float16Values.length;
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }
}
