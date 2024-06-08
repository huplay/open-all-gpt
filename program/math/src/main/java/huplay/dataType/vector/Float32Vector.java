package huplay.dataType.vector;

import huplay.MathProvider;
import huplay.dataType.DataType;

public class Float32Vector implements Vector
{
    private final float[] values;

    public Float32Vector(int size)
    {
        this.values = new float[size];
    }

    public Float32Vector(float[] values)
    {
        this.values = values;
    }

    @Override
    public DataType getFloatType()
    {
        return DataType.FLOAT_32;
    }

    @Override
    public float[] getValues()
    {
        return values;
    }

    @Override
    public float get(int index)
    {
        return values[index];
    }

    @Override
    public void set(int index, float value)
    {
        values[index] = value;
    }

    @Override
    public int size()
    {
        return values.length;
    }

    @Override
    public Vector addVector(Vector vector)
    {
        MathProvider.getMathUtility().addVector(this, vector);

        return this;
    }
}
