package math.dataType.vector;

import math.dataType.DataType;
import math.dataType.matrix.Matrix;

public interface Vector
{
    DataType getFloatType();

    float[] getValues();

    float get(int index);

    void set(int index, float value);

    int size();

    Vector add(Vector vector);

    float dotProduct(Vector vector);

    Vector multiply(float scalar);

    Vector multiply(Matrix matrix);

    Vector multiplyByTransposed(Matrix matrix);

    Matrix split(int rows);

    Vector part(int slices, int index);

    static Vector emptyVector(int size)
    {
        return emptyVector(DataType.FLOAT_32, size);
    }

    static Vector emptyVector(DataType floatType, int size)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> new Float32Vector(size);
            case FLOAT_16 -> new Float16Vector(size);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(size);
            default
                -> throw new RuntimeException("Unsupported data type at emptyVector: " + floatType);
        };
    }

    static Vector of(DataType floatType, float[] values)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> new Float32Vector(values);
            case FLOAT_16 -> new Float16Vector(values);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(values);
            default
                    -> throw new RuntimeException("Unsupported data type at of: " + floatType);
        };
    }

    static Vector of(Vector vector)
    {
        return Vector.of(vector.getFloatType(), vector.getValues());
    }

    static Vector of(DataType floatType, short[] values)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> throw new RuntimeException("Unsupported construction (Float32Vector of short[])");
            case FLOAT_16 -> new Float16Vector(values);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(values);
            default
                    -> throw new RuntimeException("Unsupported data type at of: " + floatType);
        };
    }
}
