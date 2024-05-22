package huplay.transformer;

import huplay.util.IndexedValue;
import huplay.dataType.vector.Vector;

import java.util.List;

import static huplay.AppNetworkClient.UTIL;
import static java.lang.Math.*;

public class TransformerUtil
{
    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public static float gelu(float value)
    {
        // Using a constant for sqrt(2 / PI) didn't make it faster, most likely Java optimized it
        return (float) (0.5 * value * (1 + tanh(Math.sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }

    /**
     * SwiGLU activation function
     * Original paper: <a href="https://arxiv.org/abs/2002.05202" />
     */
    public static float swiglu(float value)
    {
        return (float) (value * (1f / (1f + Math.exp(-value))));
    }

    /**
     * Standard normalization with applying normalization weights and biases
     */
    public static Vector layerNorm(Vector vector, Vector weight, Vector bias, float epsilon)
    {
        // Standard normalization
        Vector result = UTIL.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (var i = 0; i < vector.size(); i++)
        {
            result.set(i, result.get(i) * weight.get(i) + bias.get(i));
        }

        return result;
    }

    /**
     * Root Mean Square Layer Normalization (RMS)
     * Original paper: <a href="https://arxiv.org/abs/1910.07467" />
     */
    public static Vector RMSLayerNorm(Vector vector, Vector weight, float epsilon)
    {
        var size = vector.size();

        // Calculate the sum of squares
        var sum = 0f;
        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            sum += value * value;
        }

        // Calculate room mean square
        sum = 1f / sqrt(sum / size + epsilon);

        //  Normalize and scale
        Vector result = Vector.of(vector.getFloatType(), size);

        for (var i = 0; i < size; i++)
        {
            result.set(i, weight.get(i) * (sum * vector.get(i)));
        }

        return result;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static Vector softmax(Vector vector)
    {
        var max = UTIL.max(vector);

        double total = 0;
        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            var exp = exp(value - max);

            total = total + exp;
        }

        Vector ret = Vector.of(vector.getFloatType(), vector.size());

        for (var i = 0; i < vector.size(); i++)
        {
            double exp = exp(vector.get(i) - max);

            ret.set(i, (float) (exp / total));
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(List<IndexedValue> values)
    {
        var max = UTIL.max(values);

        double total = 0;
        for (var value : values)
        {
            total = total + exp(value.getValue() - max);
        }

        var ret = new float[values.size()];

        for (var i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exp(values.get(i).getValue() - max) / total);
        }

        return ret;
    }

    /**
     * Weighted random selection from list of probabilities
     */
    public static int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        var cumulativeProbabilities = new float[probabilities.length];

        for (var i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        var random = (int)(Math.random() * sum);

        var index = 0;
        for (var i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }

    public static float pow(float a, float b)
    {
        return (float)(Math.pow(a, b));
    }

    public static float sqrt(float value)
    {
        return (float)(Math.sqrt(value));
    }

    public static float cos(double value)
    {
        return (float)(Math.cos(value));
    }

    public static float sin(double value)
    {
        return (float)(Math.sin(value));
    }
}
