package math.impl.vectorApi;

import math.AbstractMathUtility;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import math.impl.standard.StandardMath;

public class VectorApiMath extends AbstractMathUtility
{
    static final VectorSpecies<Float> PROCESSOR_BLOCK = FloatVector.SPECIES_MAX;

    private static final AbstractMathUtility STANDARD_MATH = new StandardMath();

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
    public Matrix mulMatrixByScalar(Matrix matrix, float scalar)
    {
        return STANDARD_MATH.mulMatrixByScalar(matrix, scalar);
    }

    @Override
    public Vector mulVectorByMatrix(Vector vector, Matrix matrix)
    {
        return STANDARD_MATH.mulVectorByMatrix(vector, matrix);
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Matrix matrix)
    {
        return STANDARD_MATH.mulVectorByTransposedMatrix(vector, matrix);
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
    public Matrix splitVector(Vector vector, int rows)
    {
        return STANDARD_MATH.splitVector(vector, rows);
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
        return STANDARD_MATH.flattenMatrix(matrix);
    }

    @Override
    public Matrix transposeMatrix(Matrix matrix)
    {
        return STANDARD_MATH.transposeMatrix(matrix);
    }

    @Override
    public float average(Vector vector)
    {
        return STANDARD_MATH.average(vector);
    }
}
