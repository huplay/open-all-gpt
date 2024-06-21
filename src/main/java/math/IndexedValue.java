package math;

import java.util.Comparator;

/**
 * Holder of a value with the index (position of the element)
 */
public record IndexedValue(float value, int index)
{
    /**
     * Comparator for IndexedValue to achieve reverse ordering
     */
    public static class ReverseComparator implements Comparator<IndexedValue>
    {
        public int compare(IndexedValue a, IndexedValue b)
        {
            return Float.compare(b.value, a.value);
        }
    }
}

