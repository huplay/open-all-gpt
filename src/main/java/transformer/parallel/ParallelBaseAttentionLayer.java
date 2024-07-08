package transformer.parallel;

import config.Config;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import transformer.serial.BaseDecoderLayer;

import java.util.ArrayList;
import java.util.List;

public abstract class ParallelBaseAttentionLayer extends BaseDecoderLayer
{
    protected float attentionScale = 1;

    // Stored values are grouped by "kv head count" and "position"
    // At multi-head attention models (MHA, standard) the "kv head count" is the same as the "head count".
    // At grouped-query attention (GQA) "kv head count" is smaller than the "head count".
    // At multi-query attention (MQA) "kv head count" is 1.
    private final List<Matrix> storedKeys = new ArrayList<>();
    private final List<Matrix> storedValues = new ArrayList<>();

    public abstract void loadParameters();

    public abstract Matrix processParallel(Matrix inputHiddenState);

    public abstract Vector process(Vector inputHiddenState);

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);

        loadParameters();
    }

    protected void store(int head, Matrix keys, Matrix values)
    {
        initStore(keys.getInternalFloatType());

        for (var i = 0; i < keys.getRowCount(); i++)
        {
            storedKeys.get(head).addRow(keys.row(i));
            storedValues.get(head).addRow(values.row(i));
        }
    }

    protected void store(int head, Vector keys, Vector values)
    {
        initStore(keys.getFloatType());

        storedKeys.get(head).addRow(keys);
        storedValues.get(head).addRow(values);
    }

    private void initStore(DataType internalFloatType)
    {
        if (storedKeys.isEmpty())
        {
            for (var i = 0; i < kvHeadCount; i++)
            {
                storedKeys.add(Matrix.emptyMatrix(internalFloatType, 0, 0));
                storedValues.add(Matrix.emptyMatrix(internalFloatType, 0, 0));
            }
        }
    }

    protected Matrix getStoredKeys(int head)
    {
        return storedKeys.get(head);
    }

    protected Matrix getStoredValues(int head)
    {
        return storedValues.get(head);
    }

    protected int storedSize()
    {
        return storedKeys.getFirst().getRowCount();
    }

    protected void removeFirstStoredKey()
    {
        storedKeys.removeFirst();
    }

    protected void removeFirstStoredValue()
    {
        storedValues.removeFirst();
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();

        /*for (var i = 0; i < kvHeadCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }*/
    }
}
