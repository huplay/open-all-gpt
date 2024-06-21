package base;

import math.NeuralNetUtil;
import math.dataType.DataType;
import math.dataType.vector.Vector;
import org.junit.Test;

import static math.MathUtil.MATH;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NeuralNetUtilTest
{
    private final NeuralNetUtil neuralNetUtil = new TestNeuralNetUtil();

    @Test
    public void geluTest()
    {
        assertEquals(1.9545977115631104f, neuralNetUtil.gelu(2f), 1e-15f);
        assertEquals(0.005039893556386232f, neuralNetUtil.gelu(1e-2f), 1e-15f);
        assertEquals(-0.021692416f, neuralNetUtil.gelu(-0.045f), 1e-15f);
    }

    @Test
    public void swigluTest()
    {
        assertEquals(1.7615942f, neuralNetUtil.swiglu(2f), 1e-15f);
        assertEquals(0.005025f, neuralNetUtil.swiglu(1e-2f), 1e-15f);
        assertEquals(-0.021993836f, neuralNetUtil.swiglu(-0.045f), 1e-15f);
    }

    @Test
    public void layerNormTest()
    {
        var weight = Vector.of(DataType.FLOAT_32,new float[]{0.5f, -1.234f, 1e-3f, 5});
        var bias = Vector.of(DataType.FLOAT_32, new float[]{0.012f, -4.234e-3f, 1e-3f, 2});
        var epsilon = 1e-5f;

        var input = Vector.of(DataType.FLOAT_32, new float[]{1f, 2f, 3f});
        var result = MATH.layerNorm(input, weight, bias, epsilon);

        var expected = new float[]{-0.6003678f, -0.004234f, 0.0022247357f};

        assertArrayEquals(expected, result.getValues(), 1e-15f);
    }

    @Test
    public void RSMLayerNormTest()
    {
        var weight = Vector.of(DataType.FLOAT_32, new float[]{0.5f, -1.234f, 1e-3f, 5});
        var epsilon = 1e-5f;

        var input = Vector.of(DataType.FLOAT_32, new float[]{1f, 2f, 3f});
        var result = MATH.RMSLayerNorm(input, weight, epsilon);

        var expected = new float[]{0.23145477f, -1.1424607f, 0.0013887287f};

        assertArrayEquals(expected, result.getValues(), 1e-15f);
    }

    @Test
    public void RSMLayerNormWithOffsetTest()
    {
        var weight = Vector.of(DataType.FLOAT_32, new float[]{0.5f, -1.234f, 1e-3f, 5});
        var epsilon = 1e-5f;

        var input = Vector.of(DataType.FLOAT_32, new float[]{1f, 2f, 3f});
        var result = MATH.RMSLayerNorm(input, weight, epsilon, 1f);

        var expected = new float[]{0.6943643f, -0.21664163f, 1.3901174f};

        assertArrayEquals(expected, result.getValues(), 1e-15f);
    }

    @Test
    public void softmaxTest()
    {
        var values = Vector.of(DataType.FLOAT_32, new float[] {1, 2, 3, 4, 1, 2, 3});

        var result = MATH.softmax(values);

        var expected = new float[] {0.023640543f, 0.06426166f, 0.1746813f, 0.474833f, 0.023640543f, 0.06426166f, 0.1746813f};

        assertArrayEquals(expected, result.getValues(), 1e-8f);
    }
}
