package huplay.dataType.vector;

import huplay.dataType.BrainFloat16;
import huplay.dataType.DataType;

/**
 * Brain Float 16 Vector implementation (Google Brain)
 */
public class BrainFloat16Vector extends AbstractVector
{
    private final short[] values;

    public BrainFloat16Vector(int size)
    {
        this.values = new short[size];
    }

    public BrainFloat16Vector(short[] values)
    {
        this.values = values;
    }

    public BrainFloat16Vector(float[] values)
    {
        this.values = new short[values.length];

        for (var i = 0; i < values.length; i++)
        {
            this.values[i] = BrainFloat16.toShort(values[i]);
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
            floatValues[i] = BrainFloat16.toFloat32(values[i]);
        }

        return floatValues;
    }

    @Override
    public float get(int index)
    {
        return BrainFloat16.toFloat32(values[index]);
    }

    @Override
    public void set(int index, float value)
    {
        values[index] = BrainFloat16.toShort(value);
    }

    @Override
    public int size()
    {
        return values.length;
    }
}
