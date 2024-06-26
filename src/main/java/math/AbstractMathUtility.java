package math;

import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import static math.dataType.vector.Vector.emptyVector;

public abstract class AbstractMathUtility
{
    /**
     * Get the name of the math provider
     */
    public abstract String getMathProviderName();

    /**
     * Vector to vector addition
     */
    public abstract Vector addVectors(Vector vector1, Vector vector2);

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    public abstract float dotProduct(Vector vector1, Vector vector2);

    /**
     * Multiply vector by a scalar
     */
    public abstract Vector mulVectorByScalar(Vector vector, float scalar);

    /**
     * Multiply vector by matrix
     */
    public abstract Vector mulVectorByMatrix(Vector vector, Matrix matrix);

    /**
     * Multiply vector by transposed matrix
     */
    public abstract Vector mulVectorByTransposedMatrix(Vector vector, Matrix matrix);

    /**
     * Split a vector to a matrix
     */
    public abstract Matrix splitVector(Vector vector, int rows);

    public Vector joinVectors(Vector vector1, Vector vector2)
    {
        Vector vector = emptyVector(vector1.getFloatType(), vector1.size() + vector2.size());

        var i = 0;
        for (var value1 : vector1.getValues())
        {
            vector.set(i, value1);
            i++;
        }

        for (var value2 : vector2.getValues())
        {
            vector.set(i, value2);
            i++;
        }

        return vector;
    }

    /**
     * Merge the rows of a matrix to a single vector
     */
    public abstract Vector flattenMatrix(Matrix matrix);

    /**
     * Transpose a matrix
     */
    public abstract Matrix transposeMatrix(Matrix matrix);

    public byte[][] transposeByteMatrix(byte[][] matrix)
    {
        int rows = matrix.length;
        int cols = matrix[0].length;

        var transposedMatrix = new byte[cols][rows];

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                transposedMatrix[i][j] = matrix[j][i];
            }
        }

        return transposedMatrix;
    }

    /**
     * Calculate average (mean) value
     */
    public abstract float average(Vector vector);

    /**
     * Standard normalization with applying normalization weights and biases
     */
    public Vector layerNorm(Vector vector, Vector weight, Vector bias, float epsilon)
    {
        // Standard normalization
        Vector result = normalize(vector, epsilon);

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
    public Vector RMSLayerNorm(Vector vector, Vector weight, float epsilon)
    {
        return RMSLayerNorm(vector, weight, epsilon, 0f);
    }

    public Vector RMSLayerNorm(Vector vector, Vector weight, float epsilon, float bias)
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
        sum = 1f / BasicMathUtility.sqrt(sum / size + epsilon);

        //  Normalize and scale
        Vector result = emptyVector(vector.getFloatType(), size);

        for (var i = 0; i < size; i++)
        {
            result.set(i, (weight.get(i) + bias) * sum * vector.get(i));
        }

        return result;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public Vector softmax(Vector vector)
    {
        var max = max(vector);

        double total = 0;
        var exponents = new float[vector.size()];
        for (var i = 0; i < vector.size(); i++)
        {
            var exponent = BasicMathUtility.exp(vector.get(i) - max);
            exponents[i] = exponent;

            total = total + exponent;
        }

        Vector ret = emptyVector(vector.getFloatType(), vector.size());

        for (var i = 0; i < vector.size(); i++)
        {
            ret.set(i, (float) (exponents[i] / total));
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public float[] softmax(List<IndexedValue> values)
    {
        var max = max(values);

        double total = 0;
        var exponents = new float[values.size()];
        for (var i = 0; i < values.size(); i++)
        {
            var exponent = BasicMathUtility.exp(values.get(i).value() - max);
            exponents[i] = exponent;

            total = total + exponent;
        }

        var ret = new float[values.size()];

        for (var i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exponents[i] / total);
        }

        return ret;
    }

    /**
     * Weighted random selection from list of probabilities
     */
    public int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        var cumulativeProbabilities = new float[probabilities.length];

        for (var i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        var random = (int)(java.lang.Math.random() * sum);

        var index = 0;
        for (var i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }

    public float max(Vector vector)
    {
        var max = Float.NEGATIVE_INFINITY;

        for (var i = 0; i < vector.size(); i++)
        {
            var value = vector.get(i);
            if (value > max)
            {
                max = value;
            }
        }

        return max;
    }

    public float max(List<IndexedValue> vector)
    {
        var max = Float.NEGATIVE_INFINITY;

        for (var indexedValue : vector)
        {
            if (indexedValue.value() > max)
            {
                max = indexedValue.value();
            }
        }

        return max;
    }

    public Vector normalize(Vector vector, float epsilon)
    {
        var average = average(vector);
        var averageDiff = averageDiff(vector, average, epsilon);

        var norm = emptyVector(vector.getFloatType(), vector.size());

        for (var i = 0; i < vector.size(); i++)
        {
            norm.set(i, (vector.get(i) - average) / averageDiff);
        }

        return norm;
    }

    public float averageDiff(Vector values, float average, float epsilon)
    {
        var squareDiff = emptyVector(values.getFloatType(), values.size());

        for (var i = 0; i < values.size(); i++)
        {
            var diff = values.get(i) - average;
            squareDiff.set(i, diff * diff);
        }

        var averageSquareDiff = average(squareDiff);

        return (float) java.lang.Math.sqrt(averageSquareDiff + epsilon);
    }

    /**
     * Sort values to reversed order and filter out the lowest values (retain the top [count] values)
     */
    public List<IndexedValue> reverseAndFilter(float[] values, int count)
    {
        var indexedValues = new TreeSet<>(new IndexedValue.ReverseComparator());
        for (var i = 0; i < values.length; i++)
        {
            indexedValues.add(new IndexedValue(values[i], i));
        }

        var filteredValues = new ArrayList<IndexedValue>(count);

        var i = 0;
        for (var indexedValue : indexedValues)
        {
            filteredValues.add(indexedValue);
            i++;
            if (i == count) break;
        }

        return filteredValues;
    }
}
