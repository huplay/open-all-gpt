package math.impl.nd4j;

import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import math.AbstractMathUtility;
import math.impl.standard.StandardMath;
import org.nd4j.linalg.factory.Nd4j;

import static math.dataType.matrix.Matrix.emptyMatrix;

public class ND4JMath extends AbstractMathUtility
{
    private static final AbstractMathUtility STANDARD_MATH = new StandardMath();

    @Override
    public String getMathProviderName()
    {
        return "ND4j";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        try (var array1 = Nd4j.create(vector1.getValues());
             var array2 = Nd4j.create(vector2.getValues()))
        {
            return Vector.of(vector1.getFloatType(), array1.add(array2).toFloatVector());
        }
    }

    @Override
    public Matrix addMatrices(Matrix matrix1, Matrix matrix2)
    {
        return STANDARD_MATH.addMatrices(matrix1, matrix2);
    }

    @Override
    public Matrix addBroadcastVector(Matrix matrix, Vector vector)
    {
        return STANDARD_MATH.addBroadcastVector(matrix, vector);
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        try (var array1 = Nd4j.create(vector1.getValues());
             var array2 = Nd4j.create(vector2.getValues()))
        {
            return array1.mmul(array2).getFloat(0);
        }
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        try (var array = Nd4j.create(vector.getValues()))
        {
            return Vector.of(vector.getFloatType(), array.mul(scalar).toFloatVector());
        }
    }

    @Override
    public Matrix mulMatrixByScalar(Matrix matrix, float scalar)
    {
        return STANDARD_MATH.mulMatrixByScalar(matrix, scalar);
    }

    // TODO: It seems not too effective. We convert the vector to matrix and do a matrix-matrix multiplication
    public Vector mulVectorByMatrix(Vector vector, Matrix matrix)
    {
        var floatVector = new float[][] {vector.getValues()};

        var floatMatrix = new float[matrix.getRowCount()][];
        for (var i = 0; i < matrix.getRowCount(); i++)
        {
            floatMatrix[i] = matrix.row(i).getValues();
        }

        try (var array1 = Nd4j.create(floatVector);
             var array2 = Nd4j.create(floatMatrix))
        {
            return Vector.of(vector.getFloatType(), array1.mmul(array2).toFloatVector());
        }
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Matrix matrix)
    {
        var array = new float[1][vector.size()];
        array[0] = vector.getValues();

        var floatMatrix = new float[matrix.getRowCount()][];
        for (var i = 0; i < matrix.getRowCount(); i++)
        {
            floatMatrix[i] = matrix.row(i).getValues();
        }

        try (var array1 = Nd4j.create(array);
             var array2 = Nd4j.create(floatMatrix))
        {
            return Vector.of(vector.getFloatType(), array1.mmul(array2.transpose()).toFloatVector());
        }
    }

    @Override
    public Matrix mulMatrixByMatrix(Matrix matrix1, Matrix matrix2)
    {
        return STANDARD_MATH.mulMatrixByMatrix(matrix1, matrix2);
    }

    @Override
    public Matrix mulMatrixByTransposedMatrix(Matrix matrix1, Matrix matrix2)
    {
        return STANDARD_MATH.mulMatrixByTransposedMatrix(matrix1, matrix2);
    }

    @Override
    public Matrix splitVector(Vector vector, int count)
    {
        var cols = vector.size() / count;
        try (var array = Nd4j.create(vector.getValues()))
        {
            var matrix = array.reshape(count, cols).toFloatMatrix();

            var result = emptyMatrix(vector.getFloatType(), matrix.length, cols);
            for (var i = 0; i < matrix.length; i++)
            {
                result.setRow(i, Vector.of(vector.getFloatType(), matrix[i]));
            }

            return result;
        }
    }

    @Override
    public Vector partitionVector(Vector vector, int parts, int index)
    {
        return STANDARD_MATH.partitionVector(vector, parts, index);
    }

    @Override
    public Matrix partitionMatrix(Matrix vector, int parts, int index)
    {
        return STANDARD_MATH.partitionMatrix(vector, parts, index);
    }

    @Override
    public Vector flattenMatrix(Matrix matrix)
    {
        var size = (long) matrix.getRowCount() * matrix.getColCount();

        var floatMatrix = new float[matrix.getRowCount()][];
        for (var i = 0; i < matrix.getRowCount(); i++)
        {
            floatMatrix[i] = matrix.row(i).getValues();
        }

        try (var array = Nd4j.create(floatMatrix))
        {
            return Vector.of(matrix.row(0).getFloatType(), array.reshape(size).toFloatVector());
        }
    }

    @Override
    public Matrix transposeMatrix(Matrix matrix)
    {
        return STANDARD_MATH.transposeMatrix(matrix);
    }

    @Override
    public float average(Vector vector)
    {
        try (var array = Nd4j.create(vector.getValues()))
        {
            return array.meanNumber().floatValue();
        }
    }
}