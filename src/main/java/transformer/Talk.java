package transformer;

import tokenizer.Tokenizer;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static ui.ConsoleUtil.input;

public class Talk
{
    public static void talk(PrintStream OUT, TransformerFlow flow)
    {
        var pos = 0;
        var lastToken = flow.getEndOfTextToken();
        var tokenizer = flow.getTokenizer();

        while (true)
        {
            // Read the input text
            var inputText = input(OUT, "\n\nInput text: ");

            var inputTokens = new ArrayList<Integer>();

            if (inputText == null) break;
            else if (inputText.equals("+"))
            {
                // If the input is "+", continue the generation as usual (adding the last output to the input)
                inputTokens.add(lastToken);
            }
            else
            {
                if (inputText.startsWith("+"))
                {
                    // Input starts with "+" is a request to continue the same session
                    // Remove the "+", and don't clear the position and stored values
                    inputText = inputText.substring(1);
                }
                else
                {
                    // Clear the transformer's stored values
                    pos = 0;
                    flow.clear();
                }

                // Convert the input text into list of tokens
                inputTokens.addAll(tokenizer.encode(inputText));

                // Display the coloured version of the input to show the tokens
                var split = tokenizer.split(inputText);
                OUT.print("            ");
                for (var i = 0; i < split.size(); i++)
                {
                    var color = i % 2 == 0 ? "\033[32m" : "\033[33m";
                    OUT.print(color + split.get(i).getText().replace("\n", "").replace("\r", ""));
                }
                OUT.println("\033[0m");
            }

            // Use the Transformer
            var outputTokens = flow.process(inputTokens, pos);

            // Convert the output to text and print it
            var response = tokenizer.decode(outputTokens);
            print(OUT, response, outputTokens, tokenizer);

            pos += outputTokens.size();
            lastToken = outputTokens.getLast();
        }
    }

    private static void print(PrintStream OUT, String response, List<Integer> outputTokens, Tokenizer tokenizer)
    {
        // The response was printed token by token, but for multi-token characters only "�" will be displayed

        // Here we recreate the token by token decoded response (which wasn't returned)
        var tokenByTokenResponse = new StringBuilder();
        for (var token: outputTokens)
        {
            tokenByTokenResponse.append(tokenizer.decode(Collections.singletonList(token)));
        }

        // If the token by token decoded result is different to the final decoded result, print the corrected version
        if ( ! tokenByTokenResponse.toString().equals(response))
        {
            OUT.print("\nCorrected unicode response:\n" + response);
        }
    }
}
