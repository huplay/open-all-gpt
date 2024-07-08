package transformer.parallel;

import config.Config;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import network.info.DecoderBlockType;
import parameters.ParameterStore;
import transformer.TransformerType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class ParallelAbstractTransformer extends ParameterStore
{
    protected int decoderCount;

    protected final Map<Integer, ParallelBaseAttentionLayer> attentionLayers = new HashMap<>();
    protected final Map<Integer, ParallelBaseNeuralNetLayer> neuralNetLayers = new HashMap<>();

    public abstract void loadParameters();

    public abstract Matrix preProcessInputTokens(int startPos, List<Integer> tokenIds);

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
                var attentionLayer = TransformerType.getParallelAttentionLayer(config.getTransformerType());
                attentionLayer.init(config, decoderId);
                attentionLayers.put(decoderId, attentionLayer);
            }
            case NEURAL_NET_LAYER ->
            {
                var neuralNetLayer = TransformerType.getParallelNeuralNetLayer(config.getTransformerType());
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
        for (ParallelBaseAttentionLayer attentionLayer : attentionLayers.values())
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

        for (ParallelBaseAttentionLayer attentionLayer : attentionLayers.values())
        {
            parameterSize += attentionLayer.getParameterSize();
        }

        for (ParallelBaseNeuralNetLayer neuralNetLayer : neuralNetLayers.values())
        {
            parameterSize += neuralNetLayer.getParameterSize();
        }

        return parameterSize;
    }

    @Override
    public long getParameterByteSize()
    {
        var parameterByteSize = super.getParameterByteSize();

        for (ParallelBaseAttentionLayer attentionLayer : attentionLayers.values())
        {
            parameterByteSize += attentionLayer.getParameterByteSize();
        }

        for (ParallelBaseNeuralNetLayer neuralNetLayer : neuralNetLayers.values())
        {
            parameterByteSize += neuralNetLayer.getParameterByteSize();
        }

        return parameterByteSize;
    }

    public ParallelBaseAttentionLayer getAttentionLayer(int decoderId)
    {
        return attentionLayers.get(decoderId);
    }

    public ParallelBaseNeuralNetLayer getNeuralNetLayer(int decoderId)
    {
        return neuralNetLayers.get(decoderId);
    }

    public boolean isLastDecoder(int i)
    {
        return i == decoderCount - 1;
    }
}
