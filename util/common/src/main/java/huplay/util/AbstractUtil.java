package huplay.util;

import huplay.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import static huplay.dataType.vector.Vector.emptyVector;

public abstract class AbstractUtil implements Utility
{
    @Override
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

    @Override
    public float max(List<IndexedValue> vector)
    {
        var max = Float.NEGATIVE_INFINITY;

        for (var indexedValue : vector)
        {
            if (indexedValue.getValue() > max)
            {
                max = indexedValue.getValue();
            }
        }

        return max;
    }

    @Override
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

    @Override
    public float averageDiff(Vector values, float average, float epsilon)
    {
        var squareDiff = emptyVector(values.getFloatType(), values.size());

        for (var i = 0; i < values.size(); i++)
        {
            var diff = values.get(i) - average;
            squareDiff.set(i, diff * diff);
        }

        var averageSquareDiff = average(squareDiff);

        return (float) Math.sqrt(averageSquareDiff + epsilon);
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
