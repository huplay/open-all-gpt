package huplay.transformer;

import huplay.config.Config;

public abstract class BaseDecoderLayer extends ParameterStore
{
    protected int decoderId;
    protected int hiddenSize;
    protected int headCount;
    protected int headSize;
    protected boolean lastDecoder;
    protected float epsilon;

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
    }

    @Override
    protected String formatName(String name)
    {
        if (config.getParameterNameOverrides() != null)
        {
            var override = config.getParameterNameOverrides().get(name);
            if (override != null)
            {
                name = override;
            }
        }

        var decoder = "" + decoderId;
        name = name.replace("{decoderId}", decoder);

        var formattedName = config.getDecoderNameFormat().replace("{decoderId}", decoder);
        formattedName = formattedName.replace("{name}", name);

        if (config.getParameterNameOverrides() != null)
        {
            var override = config.getParameterNameOverrides().get(formattedName);
            if (override != null)
            {
                formattedName = override;
            }
        }

        return formattedName;
    }
}
