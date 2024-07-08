package math.dataType.matrix;

import static math.MathUtil.MATH;
import math.dataType.vector.Vector;

public abstract class AbstractMatrix implements Matrix
{
    @Override
    public Vector flatten()
    {
        return MATH.flattenMatrix(this);
    }

    @Override
    public Matrix add(Matrix matrix)
    {
        return MATH.addMatrices(this, matrix);
    }

    @Override
    public Matrix addBroadcast(Vector vector)
    {
        return MATH.addBroadcastVector(this, vector);
    }

    @Override
    public Matrix multiply(Matrix matrix)
    {
        return MATH.mulMatrixByMatrix(this, matrix);
    }

    @Override
    public Matrix multiplyByTransposed(Matrix matrix)
    {
        return MATH.mulMatrixByTransposedMatrix(this, matrix);
    }

    @Override
    public Matrix multiply(float scalar)
    {
        return MATH.mulMatrixByScalar(this, scalar);
    }

    @Override
    public Matrix part(int parts, int index)
    {
        return MATH.partitionMatrix(this, parts, index);
    }
}
