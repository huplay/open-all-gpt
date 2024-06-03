package huplay.transformer;

import huplay.config.Config;
import huplay.network.info.DecoderBlockType;
import huplay.IndexedValue;
import huplay.dataType.vector.Vector;
import huplay.parameters.ParameterStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static huplay.MathUtilProvider.MATH;

public abstract class BaseTransformer extends ParameterStore
{
    protected int decoderCount;
    protected int hiddenSize;
    protected int tokenCount;
    protected int embeddingCount;
    protected int contextSize;
    protected float epsilon;

    protected final Map<Integer, BaseAttentionLayer> attentionLayers = new HashMap<>();
    protected final Map<Integer, BaseNeuralNetLayer> neuralNetLayers = new HashMap<>();

    public abstract void loadParameters();

    public abstract Vector preProcessToken(int pos, int token);

    public abstract int generateToken(Vector hiddenState, int topK);

    public void init(Config config)
    {
        super.init(config);
        this.decoderCount = config.getDecoderCount();
        this.hiddenSize = config.getHiddenSize();
        this.tokenCount = config.getTokenCount();
        this.embeddingCount = config.getTokenCount();
        this.contextSize = config.getContextSize();
        this.epsilon = config.getEpsilon();

        loadParameters();
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

    public Integer processToken(int pos, int token, int topK, boolean isInputOnly)
    {
        Vector hiddenState = processTokenMain(pos, token, isInputOnly);

        if (isInputOnly)
        {
            return null; // At input processing we don't have to generate output token
        }
        else
        {
            return generateToken(hiddenState, topK);
        }
    }

    public Vector processTokenMain(int pos, int token, boolean isInputOnly)
    {
        Vector hiddenState = preProcessToken(pos, token);

        for (var i = 0; i < attentionLayers.size(); i++)
        {
            // Attention layer
            hiddenState = attentionLayers.get(i).process(hiddenState, isInputOnly);

            if (isInputOnly && i == attentionLayers.size() - 1) // During input token processing at the last decoder...
                return null; // ...we don't need the result (only the stored state at attention), unnecessary to do the rest

            // Neural net layer
            hiddenState = neuralNetLayers.get(i).process(hiddenState);
        }

        return hiddenState;
    }

    protected int selectBestToken(float[] logits, int topK)
    {
        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = MATH.reverseAndFilter(logits, topK);

        // Convert the logits to probabilities
        float[] probabilities = MATH.softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = MATH.weightedRandomPick(probabilities);

        // Lookup the token id
        return orderedLogits.get(index).getIndex();
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

    public BaseAttentionLayer getAttentionLayers(int decoderId)
    {
        return attentionLayers.get(decoderId);
    }

    public BaseNeuralNetLayer getNeuralNetLayers(int decoderId)
    {
        return neuralNetLayers.get(decoderId);
    }
}
