package math.impl.standard;

import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import math.AbstractMathUtility;

import static math.dataType.matrix.Matrix.emptyMatrix;
import static math.dataType.vector.Vector.emptyVector;

public class StandardMath extends AbstractMathUtility
{
    @Override
    public String getMathProviderName()
    {
        return "Standard";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        Vector ret = emptyVector(vector1.getFloatType(), vector1.size());

        for (int i = 0; i < vector1.size(); i++)
        {
            ret.set(i, vector1.get(i) + vector2.get(i));
        }

        return ret;
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        float sum = 0;

        for (int i = 0; i < vector1.size(); i++)
        {
            sum = sum + vector1.get(i) * vector2.get(i);
        }

        return sum;
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        Vector ret = emptyVector(vector.getFloatType(), vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            ret.set(i, vector.get(i) * scalar);
        }

        return ret;
    }

    @Override
    public Vector mulVectorByMatrix(Vector vector, Matrix matrix)
    {
        Vector ret = emptyVector(vector.getFloatType(), matrix.getColCount());

        for (int col = 0; col < matrix.getColCount(); col++)
        {
            float sum = 0;

            for (int i = 0; i < vector.size(); i++)
            {
                sum = sum + vector.get(i) * matrix.getValue(i, col);
            }

            ret.set(col, sum);
        }

        return ret;
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Matrix matrix)
    {
        Vector ret = emptyVector(vector.getFloatType(), matrix.getRowCount());

        for (int i = 0; i < matrix.getRowCount(); i++)
        {
            ret.set(i, dotProduct(vector, matrix.row(i)));
        }

        return ret;
    }

    @Override
    public Matrix splitVector(Vector vector, int rows)
    {
        int cols = vector.size() / rows;
        var matrix = emptyMatrix(vector.getFloatType(), rows, cols);

        int segment = 0;
        int col = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            matrix.setValue(segment, col, value);

            if (col == cols - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return matrix;
    }

    @Override
    public Vector flattenMatrix(Matrix matrix)
    {
        Vector ret = emptyVector(matrix.getInternalFloatType(), matrix.getRowCount() * matrix.getColCount());

        int i = 0;

        for (Vector row : matrix.getVectorArray())
        {
            for (int j = 0; j < row.size(); j++)
            {
                float value = row.get(j);
                ret.set(i, value);
                i++;
            }
        }

        return ret;
    }

    public Matrix transposeMatrix(Matrix matrix)
    {
        int rows = matrix.getRowCount();
        int cols = matrix.getColCount();

        var transposedMatrix = emptyMatrix(matrix.getInternalFloatType(), cols, rows);

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                transposedMatrix.setValue(i, j, matrix.getValue(j, i));
            }
        }

        return transposedMatrix;
    }

    @Override
    public float average(Vector vector)
    {
        double sum = 0;

        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            sum = sum + value;
        }

        return (float) sum / vector.size();
    }
}
