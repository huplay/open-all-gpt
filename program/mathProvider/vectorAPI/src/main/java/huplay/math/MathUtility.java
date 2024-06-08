package huplay.math;

import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static huplay.dataType.matrix.Matrix.emptyMatrix;
import static huplay.dataType.vector.Vector.emptyVector;

public class MathUtility extends AbstractMathUtility
{
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_MAX;

    @Override
    public String getMathProviderName()
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

    // TODO: Vector-API isn't used
    @Override
    public void addVector(Vector vector1, Vector vector2)
    {
        for (int i = 0; i < vector1.size(); i++)
        {
            vector1.set(i, vector1.get(i) + vector2.get(i));
        }
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
    public Vector mulVectorByMatrix(Vector vector, Matrix matrix)
    {
        var ret = emptyVector(matrix.getRowCount());

        for (var col = 0; col < matrix.getColCount(); col++)
        {
            float sum = 0;

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
        var ret = emptyVector(matrix.getRowCount());

        for (var col = 0; col < matrix.getRowCount(); col++)
        {
            ret.set(col, dotProduct(vector, matrix.row(col)));
        }

        return ret;
    }

    @Override
    // TODO: Vector-api isn't used
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
    // TODO: Vector-api isn't used
    public Vector flattenMatrix(Matrix matrix)
    {
        var ret = emptyVector(matrix.getRowCount() * matrix.getColCount());

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

    @Override
    // TODO: Vector-api isn't used
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