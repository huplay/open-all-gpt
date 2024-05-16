package huplay.transformer;

import huplay.config.Config;
import huplay.network.info.DecoderBlockType;
import huplay.util.IndexedValue;
import huplay.util.Vector;

import java.util.ArrayList;
import java.util.List;

import static huplay.AppNetworkClient.UTIL;
import static huplay.transformer.TransformerUtil.weightedRandomPick;
import static huplay.config.ParameterType.TOKEN_EMBEDDINGS;

public abstract class BaseTransformer extends ParameterStore
{
    protected int decoderCount;
    protected int hiddenSize;
    protected int tokenCount;
    protected int embeddingCount;
    protected int contextSize;
    protected float epsilon;

    protected final List<BaseAttentionLayer> attentionLayers = new ArrayList<>();
    protected final List<BaseNeuralNetLayer> neuralNetLayers = new ArrayList<>();

    public abstract void loadParameters();

    public abstract Vector preProcessToken(int pos, int token);

    public abstract int generateToken(Vector hiddenState, int topK);

    public void init(Config config)
    {
        this.config = config;
        this.reader = config.getReader();
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
                attentionLayers.add(attentionLayer);
            }
            case NEURAL_NET_LAYER ->
            {
                var neuralNetLayer = TransformerType.getNeuralNetLayer(config.getTransformerType());
                neuralNetLayer.init(config, decoderId);
                neuralNetLayers.add(neuralNetLayer);
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

    protected int determineOutputToken(Vector hiddenState, int topK)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(TOKEN_EMBEDDINGS)).getFloat32Values();

        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = UTIL.reverseAndFilter(logits, topK);

        // Convert the logits to probabilities
        float[] probabilities = TransformerUtil.softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = weightedRandomPick(probabilities);

        // Lookup the token id
        return orderedLogits.get(index).getIndex();
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        for (BaseAttentionLayer attentionLayer : attentionLayers)
        {
            attentionLayer.clear();
        }
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

        if (config.getParameterNaming() != null)
        {
            name = config.getParameterNaming().replace("{name}", name);
        }

        return name;
    }

    @Override
    public long getParameterSize()
    {
        var parameterSize = super.getParameterSize();

        for (BaseAttentionLayer attentionLayer : attentionLayers)
        {
            parameterSize += attentionLayer.getParameterSize();
        }

        for (BaseNeuralNetLayer neuralNetLayer : neuralNetLayers)
        {
            parameterSize += neuralNetLayer.getParameterSize();
        }

        return parameterSize;
    }

    @Override
    public long getParameterByteSize()
    {
        var parameterByteSize = super.getParameterByteSize();

        for (BaseAttentionLayer attentionLayer : attentionLayers)
        {
            parameterByteSize += attentionLayer.getParameterByteSize();
        }

        for (BaseNeuralNetLayer neuralNetLayer : neuralNetLayers)
        {
            parameterByteSize += neuralNetLayer.getParameterByteSize();
        }

        return parameterByteSize;
    }

    // Getters
    public List<BaseAttentionLayer> getAttentionLayers() {return attentionLayers;}
    public List<BaseNeuralNetLayer> getNeuralNetLayers() {return neuralNetLayers;}
}
