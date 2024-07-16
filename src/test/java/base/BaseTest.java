package base;

import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import static math.dataType.matrix.Matrix.emptyMatrix;
import static math.dataType.vector.Vector.emptyVector;
import static org.junit.Assert.*;

public class BaseTest
{
    protected Vector createVector(float... values)
    {
        var vector = emptyVector(DataType.FLOAT_32, values.length);

        for (var i = 0; i < values.length; i ++)
        {
            vector.set(i, values[i]);
        }

        return vector;
    }

    protected Matrix createMatrix(float[][] values)
    {
        var rows = values.length;
        var cols = values[0].length;
        var matrix = emptyMatrix(DataType.FLOAT_32, rows, cols);

        for (var row = 0; row < rows; row ++)
        {
            matrix.setRow(row, createVector(values[row]));
        }

        return matrix;
    }

    protected void assertVectorEquals(float[] expected, Vector actual, float delta)
    {
        if (expected == null)
        {
            assertNull(actual);
        }
        else
        {
            assertNotNull(actual);

            assertEquals(expected.length, actual.size());

            for (var i = 0; i < expected.length; i++)
            {
                assertEquals(expected[i], actual.get(i), delta);
            }
        }
    }

    protected void assertMatrixEquals(float[][] expected, Matrix actual, float delta)
    {
        if (expected == null)
        {
            assertNull(actual);
        }
        else
        {
            assertNotNull(actual);

            assertEquals(expected.length, actual.getRowCount());

            for (var i = 0; i < expected.length; i++)
            {
                assertVectorEquals(expected[i], actual.row(i), delta);
            }
        }
    }
}
