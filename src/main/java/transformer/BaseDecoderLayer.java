package transformer;

import config.Config;
import parameters.ParameterStore;

public abstract class BaseDecoderLayer extends ParameterStore
{
    protected int decoderId;
    protected int hiddenSize;
    protected int headCount;
    protected int headSize;
    protected boolean lastDecoder;
    protected float epsilon;
    protected int contextSize;

    protected int kvHeadCount;
    protected int headPerKvHead;
    protected int kvSize;

    public void init(Config config, int decoderId)
    {
        super.init(config);
        this.reader = config.getReader();
        this.decoderId = decoderId;
        this.hiddenSize = config.getHiddenSize();
        this.headCount = config.getHeadCount();
        this.headSize = config.getHeadSize();
        this.lastDecoder = (decoderId == config.getDecoderCount() - 1);
        this.epsilon = config.getEpsilon();
        this.contextSize = config.getContextSize();

        // Read the optional config for the "key/value head" count
        // (At the original attention (MHA) there was only a single kind of head. Call it "query head" from now on.)
        // If the "query head" is different to the "key/value head" count, we are using Grouped Query Attention (GQA)
        // If the "key/value head" count is 1, that is called Multi-Query Attention (MQA)
        this.kvHeadCount = config.getIntOptional("num_key_value_heads", headCount);
        this.headPerKvHead = headCount / kvHeadCount;
        this.kvSize = hiddenSize / headPerKvHead;
    }

    @Override
    protected String getFinalParameterId(String parameterId)
    {
        if (config.getParameterNameOverrides() != null)
        {
            var override = config.getParameterNameOverrides().get(parameterId);
            if (override != null)
            {
                parameterId = override;
            }
        }

        var decoder = "" + decoderId;
        parameterId = parameterId.replace("{decoderId}", decoder);

        var finalParameterId = config.getDecoderNameFormat().replace("{decoderId}", decoder);
        finalParameterId = finalParameterId.replace("{name}", parameterId);

        if (config.getParameterNameOverrides() != null)
        {
            var override = config.getParameterNameOverrides().get(finalParameterId);
            if (override != null)
            {
                finalParameterId = override;
            }
        }

        return finalParameterId;
    }
}
