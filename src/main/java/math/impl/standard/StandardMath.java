package math.impl.standard;

import app.IdentifiedException;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import math.AbstractMathUtility;

import java.util.Arrays;

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
        var ret = emptyVector(vector1.getFloatType(), vector1.size());

        for (var i = 0; i < vector1.size(); i++)
        {
            ret.set(i, vector1.get(i) + vector2.get(i));
        }

        return ret;
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        var sum = 0f;

        for (var i = 0; i < vector1.size(); i++)
        {
            sum = sum + vector1.get(i) * vector2.get(i);
        }

        return sum;
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        var result = emptyVector(vector.getFloatType(), vector.size());

        for (var i = 0; i < vector.size(); i++)
        {
            result.set(i, vector.get(i) * scalar);
        }

        return result;
    }

    @Override
    public Vector mulVectorByMatrix(Vector vector, Matrix matrix)
    {
        if (vector.size() != matrix.getRowCount())
        {
            throw new IdentifiedException("Vector and matrix shape is incompatible at multiplication. " +
                    "Vector size: " + vector.size() + ", matrix shape: " + matrix.getRowCount() + ", " + matrix.getRowCount());
        }

        var ret = emptyVector(vector.getFloatType(), matrix.getColCount());

        for (var col = 0; col < matrix.getColCount(); col++)
        {
            var sum = 0f;

            for (var i = 0; i < vector.size(); i++)
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
        if (vector.size() != matrix.getColCount())
        {
            var thread = Thread.currentThread();
            var stackTrace = thread.getStackTrace();

            throw new IdentifiedException("Vector and matrix shape is incompatible at multiplication (transposed). " +
                    "Vector size: " + vector.size() + ", matrix shape: " + matrix.getRowCount() + ", " + matrix.getColCount() +
                    " Stack trace: " + Arrays.toString(stackTrace));
        }

        var ret = emptyVector(vector.getFloatType(), matrix.getRowCount());

        for (var col = 0; col < matrix.getRowCount(); col++)
        {
            ret.set(col, dotProduct(vector, matrix.row(col)));
        }

        return ret;
    }

    @Override
    public Matrix splitVector(Vector vector, int rows)
    {
        var cols = vector.size() / rows;
        var matrix = emptyMatrix(vector.getFloatType(), rows, cols);

        var segment = 0;
        var col = 0;
        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
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
        var ret = emptyVector(matrix.getInternalFloatType(), matrix.getRowCount() * matrix.getColCount());

        var i = 0;

        for (var row : matrix.getVectorArray())
        {
            for (var j = 0; j < row.size(); j++)
            {
                var value = row.get(j);
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
        var sum = 0f;

        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            sum = sum + value;
        }

        return sum / vector.size();
    }
}
