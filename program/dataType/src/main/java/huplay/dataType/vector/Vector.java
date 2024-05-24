package huplay.dataType.vector;

import huplay.dataType.FloatType;

public interface Vector
{
    FloatType getFloatType();

    float[] getValues();

    float get(int index);

    void set(int index, float value);

    int size();

    static Vector emptyVector(int size)
    {
        return emptyVector(FloatType.FLOAT_32, size);
    }

    static Vector emptyVector(FloatType floatType, int size)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> new Float32Vector(size);
            case FLOAT_16 -> new Float16Vector(size);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(size);
        };
    }

    static Vector of(FloatType floatType, float[] values)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> new Float32Vector(values);
            case FLOAT_16 -> new Float16Vector(values);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(values);
        };
    }

    static Vector of(FloatType floatType, short[] values)
    {
        return switch (floatType)
        {
            case FLOAT_32 -> throw new RuntimeException("Unsupported construction (Float32Vector of short[])");
            case FLOAT_16 -> new Float16Vector(values);
            case BRAIN_FLOAT_16 -> new BrainFloat16Vector(values);
        };
    }
}
