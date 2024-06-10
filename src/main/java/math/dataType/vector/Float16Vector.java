package math.dataType.vector;

import math.dataType.Float16;
import math.dataType.DataType;

/**
 * Float 16 Vector implementation
 */
public class Float16Vector extends AbstractVector
{
    private final short[] values;

    public Float16Vector(int size)
    {
        this.values = new short[size];
    }

    public Float16Vector(short[] values)
    {
        this.values = values;
    }

    public Float16Vector(float[] values)
    {
        this.values = new short[values.length];

        for (var i = 0; i < values.length; i++)
        {
            this.values[i] = Float16.toShort(values[i]);
        }
    }

    @Override
    public DataType getFloatType()
    {
        return DataType.FLOAT_32;
    }

    @Override
    public float[] getValues()
    {
        var floatValues = new float[values.length];

        for (var i = 0; i < values.length; i++)
        {
            floatValues[i] = Float16.toFloat32(values[i]);
        }

        return floatValues;
    }

    @Override
    public float get(int index)
    {
        return Float16.toFloat32(values[index]);
    }

    @Override
    public void set(int index, float value)
    {
        values[index] = Float16.toShort(value);
    }

    @Override
    public int size()
    {
        return values.length;
    }
}
