package huplay;

import huplay.dataType.matrix.MatrixType;
import huplay.math.MathUtility;
import org.junit.Test;

import static huplay.dataType.matrix.Matrix.emptyMatrix;
import static org.junit.Assert.assertEquals;

public class UtilTest extends BaseTest
{
    private static final MathUtility UTIL = new MathUtility();

    @Test
    public void addVectorsTest()
    {
        var vector1 = createVector(1, 2, 3, 4);
        var vector2 = createVector(4, 5, 6, 7);
        var expectedResult = new float[]{5, 7, 9, 11};

        assertVectorEquals(expectedResult, UTIL.addVectors(vector1, vector2), 0);
    }

    @Test
    public void mulVectorByScalarTest()
    {
        var vector = createVector(5, 6, 7, 8);
        var expectedResult = new float[]{15, 18, 21, 24};

        assertVectorEquals(expectedResult, UTIL.mulVectorByScalar(vector, 3), 0);
    }

    @Test
    public void dotProductTest()
    {
        var vector1 = createVector(5, 6, 7, 8);
        var vector2 = createVector(4, 5, 6, 7);

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, UTIL.dotProduct(vector1, vector2), 0);
    }

    @Test
    public void mulVectorByMatrixTest()
    {
        var vector = createVector(2, 5, 1, 8);

        var vector1 = createVector(1, 0, 2, 0);
        var vector2 = createVector(0, 3, 0, 4);
        var vector3 = createVector(0, 0, 5, 0);
        var vector4 = createVector(6, 0, 0, 7);

        var matrix = emptyMatrix(MatrixType.VECTOR_ARRAY_FLOAT_32, 4, 4);
        matrix.setRow(0, vector1);
        matrix.setRow(1, vector2);
        matrix.setRow(2, vector3);
        matrix.setRow(3, vector4);

        var expectedResult = new float[]{4, 47, 5, 68};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(vector, matrix), 0);
    }

    @Test
    public void mulVectorByTransposedMatrixTest()
    {
        var vector = createVector(5, 6, 7, 8);

        var vector1 = createVector(1, 4, 7, 10);
        var vector2 = createVector(2, 5, 8, 11);
        var vector3 = createVector(3, 6, 9, 12);

        var matrix = emptyMatrix(MatrixType.VECTOR_ARRAY_FLOAT_32, 3, 4);
        matrix.setRow(0, vector1);
        matrix.setRow(1, vector2);
        matrix.setRow(2, vector3);

        var expectedResult = new float[]{5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(vector, matrix), 0);
    }

    @Test
    public void splitVectorTest()
    {
        var vector = createVector(1, 2, 3, 4, 5, 6);
        var expectedResult = new float[][]{{1, 2}, {3, 4}, {5, 6}};

        assertMatrixEquals(expectedResult, UTIL.splitVector(vector, 3), 0);
    }

    @Test
    public void flattenMatrixTest()
    {
        var vector1 = createVector(1, 2);
        var vector2 = createVector(3, 4);
        var vector3 = createVector(5, 6);

        var matrix = emptyMatrix(MatrixType.VECTOR_ARRAY_FLOAT_32, 3, 2);
        matrix.setRow(0, vector1);
        matrix.setRow(1, vector2);
        matrix.setRow(2, vector3);

        var expectedResult = new float[]{1, 2, 3, 4, 5, 6};

        assertVectorEquals(expectedResult, UTIL.flattenMatrix(matrix), 0);
    }

    @Test
    public void averageTest()
    {
        var vector = createVector(1, 2, 3, 4, 5, 6);

        assertEquals(3.5f, UTIL.average(vector), 0);
    }
}
