package transformer.serial;

import config.Config;
import math.dataType.vector.Vector;
import network.info.DecoderBlockType;
import parameters.ParameterStore;
import transformer.TransformerType;

import java.util.HashMap;
import java.util.Map;

public abstract class AbstractTransformer extends ParameterStore
{
    protected int decoderCount;

    protected final Map<Integer, BaseAttentionLayer> attentionLayers = new HashMap<>();
    protected final Map<Integer, BaseNeuralNetLayer> neuralNetLayers = new HashMap<>();

    public abstract void loadParameters();

    public abstract Vector preProcessToken(int pos, int tokenId);

    public abstract int generateToken(Vector hiddenState, int topK);

    public void init(Config config)
    {
        super.init(config);
        this.decoderCount = config.getDecoderCount();
    }

    public void initDecoders()
    {
        for (var i = 0; i < decoderCount; i++)
        {
            initDecoderLayer(config, i, DecoderBlockType.ATTENTION_LAYER);
            initDecoderLayer(config, i, DecoderBlockType.NEURAL_NET_LAYER);
        }
    }

    public void initDecoderLayer(Config config, int decoderId, DecoderBlockType decoderLayerType)
    {
        switch (decoderLayerType)
        {
            case ATTENTION_LAYER ->
            {
                var attentionLayer = TransformerType.getAttentionLayer(config.getTransformerType());
                attentionLayer.init(config, decoderId);
                attentionLayers.put(decoderId, attentionLayer);
            }
            case NEURAL_NET_LAYER ->
            {
                var neuralNetLayer = TransformerType.getNeuralNetLayer(config.getTransformerType());
                neuralNetLayer.init(config, decoderId);
                neuralNetLayers.put(decoderId, neuralNetLayer);
            }
        }
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        for (BaseAttentionLayer attentionLayer : attentionLayers.values())
        {
            attentionLayer.clear();
        }
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

        if (config.getParameterNaming() != null)
        {
            parameterId = config.getParameterNaming().replace("{name}", parameterId);
        }

        return parameterId;
    }

    @Override
    public long getParameterSize()
    {
        var parameterSize = super.getParameterSize();

        for (BaseAttentionLayer attentionLayer : attentionLayers.values())
        {
            parameterSize += attentionLayer.getParameterSize();
        }

        for (BaseNeuralNetLayer neuralNetLayer : neuralNetLayers.values())
        {
            parameterSize += neuralNetLayer.getParameterSize();
        }

        return parameterSize;
    }

    @Override
    public long getParameterByteSize()
    {
        var parameterByteSize = super.getParameterByteSize();

        for (BaseAttentionLayer attentionLayer : attentionLayers.values())
        {
            parameterByteSize += attentionLayer.getParameterByteSize();
        }

        for (BaseNeuralNetLayer neuralNetLayer : neuralNetLayers.values())
        {
            parameterByteSize += neuralNetLayer.getParameterByteSize();
        }

        return parameterByteSize;
    }

    public BaseAttentionLayer getAttentionLayer(int decoderId)
    {
        return attentionLayers.get(decoderId);
    }

    public BaseNeuralNetLayer getNeuralNetLayer(int decoderId)
    {
        return neuralNetLayers.get(decoderId);
    }

    public boolean isLastDecoder(int i)
    {
        return i == decoderCount - 1;
    }
}
