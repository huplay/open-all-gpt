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
