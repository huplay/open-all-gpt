package huplay.dataType.matrix;

import huplay.dataType.FloatType;
import huplay.dataType.vector.Vector;

public class VectorArrayMatrix implements Matrix
{
    private final Vector[] vectorArray;
    private final FloatType internalFloatType;

    public VectorArrayMatrix(FloatType floatType, int rows, int cols)
    {
        this.vectorArray = new Vector[rows];
        this.internalFloatType = floatType;

        for (var i = 0; i < rows; i++)
        {
            vectorArray[i] = Vector.emptyVector(floatType, cols);
        }
    }

    @Override
    public Vector[] getVectorArray()
    {
        return vectorArray;
    }

    @Override
    public Vector getVector(int index)
    {
        return vectorArray[index];
    }

    @Override
    public void setVector(int index, Vector vector)
    {
        vectorArray[index] = vector;
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
    public FloatType getInternalFloatType()
    {
        return internalFloatType;
    }
}
