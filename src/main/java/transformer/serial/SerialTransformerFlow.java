package transformer.serial;

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
public class SerialTransformerFlow implements TransformerFlow
{
    private final Config config;
    private final Tokenizer tokenizer;
    private final BaseTransformer transformer;

    public SerialTransformerFlow(Config config, Tokenizer tokenizer)
    {
        this.config = config;
        this.tokenizer = tokenizer;
        this.transformer = TransformerType.getTransformer(config.getTransformerType());

        transformer.init(config);
        transformer.initDecoders();
    }

    @Override
    public String getFlowType()
    {
        return "Serial";
    }

    /**
     * Transformer token processor
     * Implements the logic how the input tokens and the generated tokens are passed to the transformer
     */
    @Override
    public List<Integer> process(List<Integer> inputTokens, int posOffset)
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
                transformer.processToken(pos + posOffset, inputTokens.get(pos), config.getTopK(), true);
            }
        }

        // Process the last input token and repeat it with the newly generated tokens
        var token = inputTokens.get(intputSize - 1);
        OUT.println(". "); // Printing something to show there is a progress

        // Use the transformer again and again to generate new tokens
        for (var pos = intputSize - 1; pos < config.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            token = transformer.processToken( pos + posOffset, token, config.getTopK(), false);

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
