package base;

import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

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
