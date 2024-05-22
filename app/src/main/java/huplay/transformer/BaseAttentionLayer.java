package huplay.transformer;

import huplay.config.Config;
import huplay.dataType.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseAttentionLayer extends BaseDecoderLayer
{
    protected float attentionDividend;

    protected final List<Vector[]> storedKeys = new ArrayList<>();
    protected final List<Vector[]> storedValues = new ArrayList<>();

    public abstract void loadParameters();

    public abstract Vector process(Vector inputHiddenState, boolean isInputOnly);

    public void init(Config config, int decoderId)
    {
        super.init(config, decoderId);

        loadParameters();
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
