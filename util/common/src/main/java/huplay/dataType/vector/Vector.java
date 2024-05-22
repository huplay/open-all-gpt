package huplay.dataType.vector;

import huplay.dataType.FloatType;

public interface Vector
{
    FloatType getFloatType();

    float[] getValues();

    float get(int index);

    void set(int index, float value);

    int size();

    static Vector of(FloatType floatType, int size)
    {
        if (floatType.equals(FloatType.FLOAT_32))
        {
            return new Float32Vector(size);
        }
        else if (floatType.equals(FloatType.FLOAT_16))
        {
            return new Float16Vector(size);
        }
        else if (floatType.equals(FloatType.BRAIN_FLOAT_16))
        {
            return new BrainFloat16Vector(size);
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    static Vector of(FloatType floatType, float[] values)
    {
        if (floatType.equals(FloatType.FLOAT_32))
        {
            return new Float32Vector(values);
        }
        else if (floatType.equals(FloatType.FLOAT_16))
        {
            return new Float16Vector(values);
        }
        else if (floatType.equals(FloatType.BRAIN_FLOAT_16))
        {
            return new BrainFloat16Vector(values);
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    static Vector of(FloatType floatType, short[] values)
    {
        if (floatType.equals(FloatType.FLOAT_32))
        {
            throw new RuntimeException("Unsupported construction (Float32Vector of short[])");
        }
        else if (floatType.equals(FloatType.FLOAT_16))
        {
            return new Float16Vector(values);
        }
        else if (floatType.equals(FloatType.BRAIN_FLOAT_16))
        {
            return new BrainFloat16Vector(values);
        }
        else
        {
            throw new RuntimeException("Unsupported vector float type: " + floatType.name());
        }
    }

    static Vector[] newVectorArray(FloatType floatType, int rows, int cols)
    {
        Vector[] matrix = new Vector[rows];

        for (int i = 0; i < rows; i++)
        {
            matrix[i] = Vector.of(floatType, cols);
        }

        return matrix;
    }
}
