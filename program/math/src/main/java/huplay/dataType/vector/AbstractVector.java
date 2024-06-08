package huplay.dataType.vector;

import huplay.MathProvider;
import huplay.dataType.matrix.Matrix;

public abstract class AbstractVector implements Vector
{
    @Override
    public Vector add(Vector vector)
    {
        return MathProvider.getMathUtility().addVectors(this, vector);
    }

    @Override
    public float dotProduct(Vector vector)
    {
        return MathProvider.getMathUtility().dotProduct(this, vector);
    }

    @Override
    public Vector multiply(float scalar)
    {
        return MathProvider.getMathUtility().mulVectorByScalar(this, scalar);
    }

    @Override
    public Vector multiply(Matrix matrix)
    {
        return MathProvider.getMathUtility().mulVectorByMatrix(this, matrix);
    }

    @Override
    public Vector multiplyByTransposed(Matrix matrix)
    {
        return MathProvider.getMathUtility().mulVectorByTransposedMatrix(this, matrix);
    }

    @Override
    public Matrix split(int rows)
    {
        return MathProvider.getMathUtility().splitVector(this, rows);
    }
}
