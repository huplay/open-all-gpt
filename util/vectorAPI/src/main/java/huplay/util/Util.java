package huplay.util;

import huplay.dataType.vector.Vector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Util extends AbstractUtil
{
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_MAX;

    @Override
    public String getUtilName()
    {
        return "Java Vector API (" + SPECIES.vectorBitSize() + " bit)";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        var result = new float[vector1.size()];

        for (var i = 0; i < vector1.size(); i += SPECIES.length())
        {
            var mask = SPECIES.indexInRange(i, vector1.size());
            var first = FloatVector.fromArray(SPECIES, vector1.getValues(), i, mask);
            var second = FloatVector.fromArray(SPECIES, vector2.getValues(), i, mask);
            first.add(second).intoArray(result, i, mask);
        }

        return Vector.of(vector1.getFloatType(), result);
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        var upperBound = SPECIES.loopBound(vector1.size());
        var sum = FloatVector.zero(SPECIES);

        var i = 0;
        for (; i < upperBound; i += SPECIES.length())
        {
            var va = FloatVector.fromArray(SPECIES, vector1.getValues(), i);
            var vb = FloatVector.fromArray(SPECIES, vector2.getValues(), i);
            sum = va.fma(vb, sum);
        }

        var result = sum.reduceLanes(VectorOperators.ADD);

        // counter "i" has an incremented value from the previous loop
        for (; i < vector1.size(); i++)
        {
            result += vector1.get(i) * vector2.get(i);
        }

        return result;
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        var result = new float[vector.size()];

        for (var i = 0; i < vector.size(); i += SPECIES.length())
        {
            var mask = SPECIES.indexInRange(i, vector.size());
            var floatVector = FloatVector.fromArray(SPECIES, vector.getValues(), i, mask);
            floatVector.mul(scalar).intoArray(result, i, mask);
        }

        return Vector.of(vector.getFloatType(), result);
    }

    @Override
    // TODO: Vector-api isn't used
    public Vector mulVectorByMatrix(Vector vector, Vector[] matrix)
    {
        var ret = Vector.of(vector.getFloatType(), matrix[0].size());

        for (var col = 0; col < matrix[0].size(); col++)
        {
            float sum = 0;

            for (var i = 0; i < vector.size(); i++)
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
        var ret = Vector.of(vector.getFloatType(), matrix.length);

        for (var col = 0; col < matrix.length; col++)
        {
            ret.set(col, dotProduct(vector, matrix[col]));
        }

        return ret;
    }

    @Override
    // TODO: Vector-api isn't used
    public Vector[] splitVector(Vector vector, int count)
    {
        var size = vector.size() / count;
        var ret = Vector.newVectorArray(vector.getFloatType(), count, size);

        var segment = 0;
        var col = 0;
        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
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
    // TODO: Vector-api isn't used
    public Vector flattenMatrix(Vector[] matrix)
    {
        var ret = Vector.of(matrix[0].getFloatType(), matrix.length * matrix[0].size());

        var i = 0;

        for (var row : matrix)
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

    @Override
    // TODO: Vector-api isn't used
    public float average(Vector vector)
    {
        float sum = 0;

        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            sum = sum + value;
        }

        return sum / vector.size();
    }
}