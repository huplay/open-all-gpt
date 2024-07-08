package transformer.parallel;

import config.Config;
import math.IndexedValue;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import java.util.List;

import static math.MathUtil.MATH;

public abstract class ParallelBaseTransformer extends ParallelAbstractTransformer
{
    protected int hiddenSize;
    protected int tokenCount;
    protected int embeddingCount;
    protected int contextSize;
    protected float epsilon;

    public void init(Config config)
    {
        super.init(config);
        this.hiddenSize = config.getHiddenSize();
        this.tokenCount = config.getTokenCount();
        this.embeddingCount = config.getTokenCount();
        this.contextSize = config.getContextSize();
        this.epsilon = config.getEpsilon();

        // Load the main parameters (used by the head and tail) - call the loadParameters of the implementation
        loadParameters();
    }

    /**
     * Main flow of processing the input tokens
     */
    public Integer processInputTokens(int startPos, List<Integer> tokenIds, int topK)
    {
        // Head of the transformer - call the preProcess of the implementation
        Matrix hiddenState = preProcessInputTokens(startPos, tokenIds);

        for (var i = 0; i < decoderCount; i++)
        {
            // Attention layer - call the attention block of the implementation
            hiddenState = getAttentionLayer(i).processParallel(hiddenState);

            // Neural net layer  - call the neural net block of the implementation
            hiddenState = getNeuralNetLayer(i).processParallel(hiddenState);
        }

        return generateToken(hiddenState.row(hiddenState.getRowCount() - 1), topK);
    }

    /**
     * Main flow of processing a generated token
     */
    public Integer processToken(int pos, int tokenId, int topK)
    {
        // Head of the transformer - call the preProcess of the implementation
        Vector hiddenState = processTokenMain(pos, tokenId);

        // Tail of the transformer - call the generateToken of the implementation
        return generateToken(hiddenState, topK);
    }

    public Vector processTokenMain(int pos, int token)
    {
        // Head of the transformer - call the preProcess of the implementation
        Vector hiddenState = preProcessToken(pos, token);

        for (var i = 0; i < decoderCount; i++)
        {
            // Attention layer - call the attention block of the implementation
            hiddenState = getAttentionLayer(i).process(hiddenState);

            // Neural net layer  - call the neural net block of the implementation
            hiddenState = getNeuralNetLayer(i).process(hiddenState);
        }

        return hiddenState;
    }

    /**
     * Selects randomly the token from the provided logits (list of probabilities), using the topK settings
     */
    protected int selectBestToken(Vector logits, int topK)
    {
        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = MATH.reverseAndFilter(logits.getValues(), topK);

        // Convert the logits to probabilities
        float[] probabilities = MATH.softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = MATH.weightedRandomPick(probabilities);

        // Lookup the token id
        return orderedLogits.get(index).index();
    }
}
