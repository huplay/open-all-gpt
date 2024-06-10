package transformer;

import config.Config;
import tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static app.AppStandaloneMain.OUT;

/**
 * Decoder-only Transformer implementation
 */
public class TransformerFlow
{
    private final Config config;
    private final Tokenizer tokenizer;
    private final BaseTransformer transformer;

    public TransformerFlow(Config config, Tokenizer tokenizer, BaseTransformer transformer)
    {
        this.config = config;
        this.tokenizer = tokenizer;
        this.transformer = transformer;

        transformer.init(config);
        transformer.initDecoders();
    }

    /**
     * Transformer token processor
     * Implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> process(List<Integer> inputTokens, int startPos)
    {
        var result = new ArrayList<Integer>();
        var intputSize = inputTokens.size();

        // Process the input tokens (except the last)
        if (intputSize == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(config.getEndOfTextToken());
            intputSize = 1;
        }
        else
        {
            // Iterating over on the input tokens (excluding the last one) and processing these by the transformer
            // We are not interested in the output of the transformer, but the inner state will be stored
            for (var pos = 0; pos < intputSize - 1; pos++)
            {
                OUT.print("."); // Printing a dot to show there is a progress
                transformer.processToken(pos + startPos, inputTokens.get(pos), config.getTopK(), true);
            }
        }

        // Process the last input token and repeat it with the newly generated tokens
        var token = inputTokens.get(intputSize - 1);
        OUT.println(". "); // Printing something to show there is a progress

        // Use the transformer again and again to generate new tokens
        for (var pos = intputSize - 1; pos < config.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            token = transformer.processToken( pos + startPos, token, config.getTopK(), false);

            // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
            OUT.print(tokenizer.decode(Collections.singletonList(token)));

            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the maximum length is reached
            if (token == config.getEndOfTextToken())
            {
                break;
            }

            // Exit if we reached the context size
            if (intputSize + result.size() + startPos >= config.getContextSize()) break;
        }

        return result;
    }

    public void clear()
    {
        transformer.clear();
    }

    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }

    public int getEndOfTextToken()
    {
        return config.getEndOfTextToken();
    }
}
