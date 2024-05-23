package huplay;

import huplay.dataType.matrix.Matrix;
import huplay.dataType.matrix.MatrixType;
import huplay.util.Util;
import huplay.dataType.vector.Vector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class UtilTest extends BaseTest
{
    private static final Util UTIL = new Util();

    @Test
    public void addVectorsTest()
    {
        var a = createVector(1, 2, 3, 4);
        var b = createVector(4, 5, 6, 7);
        var expectedResult = new float[]{5, 7, 9, 11};

        assertVectorEquals(expectedResult, UTIL.addVectors(a, b), 0);
    }

    @Test
    public void mulVectorByScalarTest()
    {
        var a = createVector(5, 6, 7, 8);
        var expectedResult = new float[]{15, 18, 21, 24};

        assertVectorEquals(expectedResult, UTIL.mulVectorByScalar(a, 3), 0);
    }

    @Test
    public void dotProductTest()
    {
        var a = createVector(5, 6, 7, 8);
        var b = createVector(4, 5, 6, 7);

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, UTIL.dotProduct(a, b), 0);
    }

    @Test
    public void mulVectorByMatrixTest()
    {
        var a = createVector(2, 5, 1, 8);

        var b1 = createVector(1, 0, 2, 0);
        var b2 = createVector(0, 3, 0, 4);
        var b3 = createVector(0, 0, 5, 0);
        var b4 = createVector(6, 0, 0, 7);
        var b = Matrix.of(new Vector[]{b1, b2, b3, b4});

        var expectedResult = new float[]{4, 47, 5, 68};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void mulVectorByTransposedMatrixTest()
    {
        var a = createVector(5, 6, 7, 8);

        var b1 = createVector(1, 4, 7, 10);
        var b2 = createVector(2, 5, 8, 11);
        var b3 = createVector(3, 6, 9, 12);
        var b = Matrix.of(new Vector[]{b1, b2, b3});

        var expectedResult = new float[]{5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
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
        var v1 = createVector(1, 2);
        var v2 = createVector(3, 4);
        var v3 = createVector(5, 6);
        var matrix = Matrix.of(new Vector[]{v1, v2, v3});

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
