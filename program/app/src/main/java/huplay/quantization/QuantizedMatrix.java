package huplay.quantization;

import huplay.IdentifiedException;
import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.matrix.VectorArrayMatrix;
import huplay.dataType.vector.Vector;

public abstract class QuantizedMatrix implements Matrix
{
    private final DataType outputFloatType;

    public QuantizedMatrix(DataType outputFloatType)
    {
        this.outputFloatType = outputFloatType;
    }

    public Matrix toDeQuantized()
    {
        var result = new VectorArrayMatrix(getInternalFloatType(), getRowCount(), getColCount());
        for (var i = 0; i < getRowCount(); i++)
        {
            result.setRow(i, getRow(i));
        }
        return result;
    }

    @Override
    public void setValue(int row, int col, float value)
    {
        throw new IdentifiedException("Unsupported operation: setValue on quantized matrix");
    }

    @Override
    public Vector getRow(int row)
    {
        var vector = Vector.emptyVector(getInternalFloatType(), getColCount());

        for (int i = 0; i < getColCount(); i++)
        {
            vector.set(i, getValue(row, i));
        }

        return vector;
    }

    @Override
    public void setRow(int row, Vector vector)
    {
        throw new IdentifiedException("Unsupported operation: setVector on quantized matrix");
    }

    @Override
    public DataType getInternalFloatType()
    {
        return outputFloatType;
    }
}
