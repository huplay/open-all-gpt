package quantization;

import app.IdentifiedException;
import math.dataType.DataType;
import math.dataType.matrix.AbstractMatrix;
import math.dataType.matrix.Matrix;
import math.dataType.matrix.VectorArrayMatrix;
import math.dataType.vector.Vector;

public abstract class QuantizedMatrix extends AbstractMatrix
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
            result.setRow(i, row(i));
        }
        return result;
    }

    @Override
    public void setValue(int rowId, int colId, float value)
    {
        throw new IdentifiedException("Unsupported operation: setValue on quantized matrix");
    }

    @Override
    public Vector row(int rowId)
    {
        var vector = Vector.emptyVector(getInternalFloatType(), getColCount());

        for (int colId = 0; colId < getColCount(); colId++)
        {
            vector.set(colId, getValue(rowId, colId));
        }

        return vector;
    }

    @Override
    public void setRow(int rowId, Vector vector)
    {
        throw new IdentifiedException("Unsupported operation: setVector on quantized matrix");
    }

    @Override
    public Vector[] getVectorArray()
    {
        var vectorArray = new Vector[getRowCount()];

        for (int i = 0; i < getRowCount(); i++)
        {
            vectorArray[i] = row(i);
        }

        return vectorArray;
    }

    @Override
    public DataType getInternalFloatType()
    {
        return outputFloatType;
    }
}
