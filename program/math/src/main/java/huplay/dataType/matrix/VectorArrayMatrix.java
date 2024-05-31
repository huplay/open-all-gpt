package huplay.dataType.matrix;

import huplay.dataType.DataType;
import huplay.dataType.vector.Vector;

public class VectorArrayMatrix implements Matrix
{
    private final Vector[] vectorArray;
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
    public float getValue(int row, int col)
    {
        return vectorArray[row].get(col);
    }

    @Override
    public void setValue(int row, int col, float value)
    {
        vectorArray[row].set(col, value);
    }

    @Override
    public Vector getRow(int row)
    {
        return vectorArray[row];
    }

    @Override
    public void setRow(int row, Vector vector)
    {
        vectorArray[row] = vector;
    }

    @Override
    public Vector[] getVectorArray()
    {
        return vectorArray;
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
