package math.impl.standard;

import app.IdentifiedException;
import math.dataType.matrix.Matrix;
import math.dataType.matrix.VectorArrayMatrix;
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
    public Matrix addMatrices(Matrix matrix1, Matrix matrix2)
    {
        var rows = matrix1.getRowCount();
        var ret = new VectorArrayMatrix(matrix1.getInternalFloatType(), rows, matrix1.getColCount());

        for (var i = 0; i < rows; i++)
        {
            ret.setRow(i, addVectors(matrix1.row(i), matrix2.row(i)));
        }

        return ret;
    }

    @Override
    public Matrix addBroadcastVector(Matrix matrix, Vector vector)
    {
        var rows = matrix.getRowCount();
        var ret = new VectorArrayMatrix(matrix.getInternalFloatType(), rows, matrix.getColCount());

        for (var i = 0; i < rows; i++)
        {
            ret.setRow(i, addVectors(matrix.row(i), vector));
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
    public Matrix mulMatrixByScalar(Matrix matrix, float scalar)
    {
        var rows = matrix.getRowCount();
        var ret = new VectorArrayMatrix(matrix.getInternalFloatType(), rows, matrix.getColCount());

        for (var i = 0; i < rows; i++)
        {
            ret.setRow(i, mulVectorByScalar(matrix.row(i), scalar));
        }

        return ret;
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
    public Matrix mulMatrixByMatrix(Matrix matrix1, Matrix matrix2)
    {
        var result = emptyMatrix(matrix1.getInternalFloatType(), matrix1.getRowCount(), matrix2.getColCount());

        for (var i = 0; i < matrix1.getRowCount(); i++)
        {
            result.setRow(i, mulVectorByMatrix(matrix1.row(i), matrix2));
        }

        return result;
    }

    @Override
    public Matrix mulMatrixByTransposedMatrix(Matrix matrix1, Matrix matrix2)
    {
        var result = emptyMatrix(matrix1.getInternalFloatType(), matrix1.getRowCount(), matrix2.getColCount());

        for (var i = 0; i < matrix1.getRowCount(); i++)
        {
            result.setRow(i, mulVectorByTransposedMatrix(matrix1.row(i), matrix2));
        }

        return result;
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
    public Vector partitionVector(Vector vector, int parts, int index)
    {
        var size = vector.size() / parts;
        var result = emptyVector(vector.getFloatType(), size);

        for (var i = 0; i < size; i++)
        {
            var value = vector.get(i + size * index);
            result.set(i, value);
        }

        return result;
    }

    @Override
    public Matrix partitionMatrix(Matrix matrix, int parts, int index)
    {
        var width = matrix.getColCount() / parts;
        var result = new VectorArrayMatrix(matrix.getInternalFloatType(), matrix.getRowCount(), width);

        for (var row = 0; row < matrix.getRowCount(); row++)
        {
            for (var i = 0; i < width; i++)
            {
                result.setValue(row, i, matrix.getValue(row, i + width * index));
            }
        }

        return result;
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
