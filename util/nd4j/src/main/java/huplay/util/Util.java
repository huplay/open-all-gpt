package huplay.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Util extends AbstractUtil
{
    @Override
    public String getUtilName()
    {
        return "ND4j";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        try (var array1 = Nd4j.create(vector1.getFloat32Values());
             var array2 = Nd4j.create(vector2.getFloat32Values()))
        {
            return new Vector(vector1.getFloatType(), array1.add(array2).toFloatVector());
        }
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        try (var array1 = Nd4j.create(vector1.getFloat32Values());
             var array2 = Nd4j.create(vector2.getFloat32Values()))
        {
            return array1.mmul(array2).getFloat(0);
        }
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        try (var array = Nd4j.create(vector.getFloat32Values()))
        {
            return new Vector(vector.getFloatType(), array.mul(scalar).toFloatVector());
        }
    }

    // TODO: It seems not too effective. We convert the vector to matrix and do a matrix-matrix multiplication
    public Vector mulVectorByMatrix(Vector vector, Vector[] matrix)
    {
        var floatVector = new float[][] {vector.getFloat32Values()};

        var floatMatrix = new float[matrix.length][];
        for (var i = 0; i < matrix.length; i++)
        {
            floatMatrix[i] = matrix[i].getFloat32Values();
        }

        try (var array1 = Nd4j.create(floatVector);
             var array2 = Nd4j.create(floatMatrix))
        {
            return new Vector(vector.getFloatType(), array1.mmul(array2).toFloatVector());
        }
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Vector[] matrix)
    {
        var array = new float[1][vector.size()];
        array[0] = vector.getFloat32Values();

        var floatMatrix = new float[matrix.length][];
        for (var i = 0; i < matrix.length; i++)
        {
            floatMatrix[i] = matrix[i].getFloat32Values();
        }

        try (var array1 = Nd4j.create(array);
             var array2 = Nd4j.create(floatMatrix))
        {
            return new Vector(vector.getFloatType(), array1.mmul(array2.transpose()).toFloatVector());
        }
    }

    @Override
    public Vector[] splitVector(Vector vector, int count)
    {
        try (var array = Nd4j.create(vector.getFloat32Values()))
        {
            var matrix = array.reshape(count, vector.size() / count).toFloatMatrix();

            var result = new Vector[matrix.length];
            for (var i = 0; i < matrix.length; i++)
            {
                result[i] = new Vector(vector.getFloatType(), matrix[i]);
            }

            return result;
        }
    }

    @Override
    public Vector flattenMatrix(Vector[] matrix)
    {
        var size = (long) matrix.length * matrix[0].size();

        var floatMatrix = new float[matrix.length][];
        for (var i = 0; i < matrix.length; i++)
        {
            floatMatrix[i] = matrix[i].getFloat32Values();
        }

        try (var array = Nd4j.create(floatMatrix))
        {
            return new Vector(matrix[0].getFloatType(), array.reshape(size).toFloatVector());
        }
    }

    @Override
    public float average(Vector vector)
    {
        try (var array = Nd4j.create(vector.getFloat32Values()))
        {
            return array.meanNumber().floatValue();
        }
    }
}