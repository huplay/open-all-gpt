package math.dataType.vector;

import static math.MathUtil.MATH;
import math.dataType.matrix.Matrix;

public abstract class AbstractVector implements Vector
{
    @Override
    public Vector add(Vector vector)
    {
        return MATH.addVectors(this, vector);
    }

    @Override
    public float dotProduct(Vector vector)
    {
        return MATH.dotProduct(this, vector);
    }

    @Override
    public Vector multiply(float scalar)
    {
        return MATH.mulVectorByScalar(this, scalar);
    }

    @Override
    public Vector multiply(Matrix matrix)
    {
        return MATH.mulVectorByMatrix(this, matrix);
    }

    @Override
    public Vector multiplyByTransposed(Matrix matrix)
    {
        return MATH.mulVectorByTransposedMatrix(this, matrix);
    }

    @Override
    public Matrix split(int rows)
    {
        return MATH.splitVector(this, rows);
    }
}
