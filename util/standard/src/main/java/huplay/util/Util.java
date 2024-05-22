package huplay.util;

import huplay.dataType.vector.Vector;

public class Util extends AbstractUtil
{
    @Override
    public String getUtilName()
    {
        return "Standard";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        Vector ret = Vector.of(vector1.getFloatType(), vector1.size());

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
        Vector ret = Vector.of(vector.getFloatType(), vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            ret.set(i, vector.get(i) * scalar);
        }

        return ret;
    }

    @Override
    public Vector mulVectorByMatrix(Vector vector, Vector[] matrix)
    {
        Vector ret = Vector.of(vector.getFloatType(), matrix[0].size());

        for (int col = 0; col < matrix[0].size(); col++)
        {
            float sum = 0;

            for (int i = 0; i < vector.size(); i++)
            {
                sum = sum + vector.get(i) * matrix[i].get(col);
            }

            ret.set(col, sum);
        }

        return ret;
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Vector[] matrix)
    {
        Vector ret = Vector.of(vector.getFloatType(), matrix.length);

        for (int i = 0; i < matrix.length; i++)
        {
            ret.set(i, dotProduct(vector, matrix[i]));
        }

        return ret;
    }

    @Override
    public Vector[] splitVector(Vector vector, int count)
    {
        int size = vector.size() / count;
        Vector[] ret = Vector.newVectorArray(vector.getFloatType(), count, size);

        int segment = 0;
        int col = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            ret[segment].set(col, value);

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return ret;
    }

    @Override
    public Vector flattenMatrix(Vector[] matrix)
    {
        Vector ret = Vector.of(matrix[0].getFloatType(), matrix.length * matrix[0].size());

        int i = 0;

        for (Vector row : matrix)
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
