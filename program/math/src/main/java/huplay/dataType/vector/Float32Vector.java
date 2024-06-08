package huplay.dataType.vector;

import huplay.dataType.DataType;

public class Float32Vector extends AbstractVector
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
}
