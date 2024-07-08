package transformer.parallel;

import config.Config;
import tokenizer.Tokenizer;
import transformer.TransformerFlow;
import transformer.TransformerType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static app.AppStandaloneMain.OUT;

/**
 * Decoder-only Transformer implementation
 */
public class ParallelTransformerFlow implements TransformerFlow
{
    private final Config config;
    private final Tokenizer tokenizer;
    private final ParallelBaseTransformer transformer;

    public ParallelTransformerFlow(Config config, Tokenizer tokenizer)
    {
        this.config = config;
        this.tokenizer = tokenizer;
        this.transformer = TransformerType.getParallelTransformer(config.getTransformerType());

        transformer.init(config);
        transformer.initDecoders();
    }

    @Override
    public String getFlowType()
    {
        return "Parallel";
    }

    /**
     * Transformer token processor
     * Implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    @Override
    public List<Integer> process(List<Integer> inputTokens, int posOffset)
    {
        var result = new ArrayList<Integer>();
        var intputSize = inputTokens.size();

        // If the input is empty, use the END_OF_TEXT token as input
        if (intputSize == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(config.getEndOfTextToken());
            intputSize = 1;
        }

        // Process the input tokens - parallel process
        Integer token = transformer.processInputTokens(posOffset, inputTokens, config.getTopK());
        result.add(token);

        // Use the transformer again and again to generate new tokens
        for (var pos = intputSize; pos < config.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            token = transformer.processToken( pos + posOffset, token, config.getTopK());

            // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
            OUT.print(tokenizer.decode(Collections.singletonList(token)));

            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the maximum length is reached
            if (token == config.getEndOfTextToken())
            {
                break;
            }

            // Exit if we reached the context size
            if (intputSize + result.size() + posOffset >= config.getContextSize()) break;
        }

        return result;
    }

    @Override
    public void clear()
    {
        transformer.clear();
    }

    @Override
    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }

    @Override
    public int getEndOfTextToken()
    {
        return config.getEndOfTextToken();
    }

    @Override
    public long getParameterSize()
    {
        return transformer.getParameterSize();
    }
}
