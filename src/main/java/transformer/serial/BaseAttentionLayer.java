package transformer.serial;

import config.Config;import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseAttentionLayer extends BaseDecoderLayer
{
    protected float attentionScale = 1;

    // Stored values are grouped by "kv head count" and "position"
    // At multi-head attention models (MHA, standard) the "kv head count" is the same as the "head count".
    // At grouped-query attention (GQA) "kv head count" is smaller than the "head count".
    // At multi-query attention (MQA) "kv head count" is 1.
    private final List<List<Vector>> storedKeys = new ArrayList<>();
    private final List<List<Vector>> storedValues = new ArrayList<>();

    public abstract void loadParameters();

    public abstract Vector process(Vector inputHiddenState, boolean isInputOnly);

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);

        for (var i = 0; i < kvHeadCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }

        loadParameters();
    }

    protected void store(int head, Vector key, Vector value)
    {
        storedKeys.get(head).add(key);
        storedValues.get(head).add(value);
    }

    @Deprecated
    protected void store(int head, Matrix keys, Matrix values)
    {
        storedKeys.get(head).add(keys.row(head));
        storedValues.get(head).add(values.row(head));
    }

    protected List<Vector> getStoredKeys(int head)
    {
        return storedKeys.get(head);
    }

    protected Vector getStoredKey(int head, int pos)
    {
        return storedKeys.get(head).get(pos);
    }

    protected List<Vector> getStoredValues(int head)
    {
        return storedValues.get(head);
    }

    protected Vector getStoredValue(int head, int pos)
    {
        return storedValues.get(head).get(pos);
    }

    protected int storedSize()
    {
        return storedKeys.getFirst().size();
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

        for (var i = 0; i < kvHeadCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }
    }
}
