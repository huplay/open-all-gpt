package math.dataType.matrix;

import math.dataType.DataType;
import math.dataType.vector.Vector;

public class VectorArrayMatrix extends AbstractMatrix
{
    private Vector[] vectorArray;
    private final DataType internalFloatType;

    public VectorArrayMatrix(DataType floatType, int rows, int cols)
    {
        this.vectorArray = new Vector[rows];
        this.internalFloatType = floatType;

        for (var i = 0; i < rows; i++)
        {
            vectorArray[i] = Vector.emptyVector(floatType, cols);
        }
    }

    @Override
    public float getValue(int rowId, int colId)
    {
        return vectorArray[rowId].get(colId);
    }

    @Override
    public void setValue(int rowId, int colId, float value)
    {
        vectorArray[rowId].set(colId, value);
    }

    @Override
    public Vector row(int rowId)
    {
        return vectorArray[rowId];
    }

    @Override
    public void setRow(int rowId, Vector vector)
    {
        vectorArray[rowId] = vector;
    }

    @Override
    public Vector[] getVectorArray()
    {
        return vectorArray;
    }

    @Override
    public void addRow(Vector vector)
    {
        var index = vectorArray.length;
        Vector[] newVectorArray = new Vector[index + 1];

        System.arraycopy(vectorArray, 0, newVectorArray, 0, vectorArray.length);
        newVectorArray[index] = vector;

        this.vectorArray = newVectorArray;
    }

    @Override
    public int getRowCount()
    {
        return vectorArray.length;
    }

    @Override
    public int getColCount()
    {
        return vectorArray[0].size();
    }

    @Override
    public DataType getInternalFloatType()
    {
        return internalFloatType;
    }
}
