package huplay.util;

import java.util.Comparator;

/**
 * Holder of a value with the index (position of the element)
 */
public class IndexedValue
{
    private final float value;
    private final int index;

    public IndexedValue(float value, int index)
    {
        this.value = value;
        this.index = index;
    }

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

    public float getValue()
    {
        return value;
    }

    public int getIndex()
    {
        return index;
    }
}

