package math.impl.vectorApi;

import app.IdentifiedException;
import math.AbstractMathUtility;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

import static math.dataType.matrix.Matrix.emptyMatrix;
import static math.dataType.vector.Vector.emptyVector;

public class VectorApiMath extends AbstractMathUtility
{
    static final VectorSpecies<Float> PROCESSOR_BLOCK = FloatVector.SPECIES_MAX;

    @Override
    public String getMathProviderName()
    {
        return "Java Vector API (" + PROCESSOR_BLOCK.vectorBitSize() + " bit)";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        var result = new float[vector1.size()];

        for (var i = 0; i < vector1.size(); i += PROCESSOR_BLOCK.length())
        {
            var mask = PROCESSOR_BLOCK.indexInRange(i, vector1.size());
            var first = FloatVector.fromArray(PROCESSOR_BLOCK, vector1.getValues(), i, mask);
            var second = FloatVector.fromArray(PROCESSOR_BLOCK, vector2.getValues(), i, mask);
            first.add(second).intoArray(result, i, mask);
        }

        return Vector.of(vector1.getFloatType(), result);
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        // Determine how many full loops can we do (vector size divided by the vector block size)
        var fullLoops = PROCESSOR_BLOCK.loopBound(vector1.size());

        // Processing the full loops using fma and reduceLanes

        var sum = FloatVector.zero(PROCESSOR_BLOCK);

        var i = 0;
        for (; i < fullLoops; i += PROCESSOR_BLOCK.length())
        {
            var va = FloatVector.fromArray(PROCESSOR_BLOCK, vector1.getValues(), i);
            var vb = FloatVector.fromArray(PROCESSOR_BLOCK, vector2.getValues(), i);
            sum = va.fma(vb, sum);
        }

        var result = sum.reduceLanes(VectorOperators.ADD);

        // variable "i" has an incremented value from the previous loop
        for (; i < vector1.size(); i++)
        {
            // This will be executed only few times, for the last elements
            // if the vector size isn't divisible by the processor block size
            result += vector1.get(i) * vector2.get(i);
        }

        return result;
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        var result = new float[vector.size()];

        for (var i = 0; i < vector.size(); i += PROCESSOR_BLOCK.length())
        {
            var mask = PROCESSOR_BLOCK.indexInRange(i, vector.size());
            var floatVector = FloatVector.fromArray(PROCESSOR_BLOCK, vector.getValues(), i, mask);
            floatVector.mul(scalar).intoArray(result, i, mask);
        }

        return Vector.of(vector.getFloatType(), result);
    }

    @Override
    // TODO: Vector-api isn't used
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
        var sum = 0f;

        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            sum = sum + value;
        }

        return sum / vector.size();
    }
}
