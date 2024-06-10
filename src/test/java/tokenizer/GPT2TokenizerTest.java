package tokenizer;

import config.TokenizerConfig;
import tokenizer.Tokenizer;
import tokenizer.gpt.GPT2Tokenizer;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class GPT2TokenizerTest
{
    private static final Tokenizer tokenizer = getTokenizer();

    @Test
    public void encoderTest()
    {
        testEncoder("Test", 14402);
        testEncoder("Test this sentence.", 14402, 428, 6827, 13);
        testEncoder("I would like 123 to try this!", 40, 561, 588, 17031, 284, 1949, 428, 0);
        testEncoder("I would like 123 to TRY this!", 40, 561, 588, 17031, 284, 7579, 56, 428, 0);

        testEncoder("é", 2634);
        testEncoder("ő", 129, 239);

        testEncoder("Szeretnék egy ékezetekkel rendelkező magyar mondatot is kipróbálni.",
                50, 89, 31229, 77, 2634, 74, 304, 1360, 38251, 365, 89, 316, 988, 7750, 9851, 417, 365, 89, 129,
                239, 2153, 88, 283, 285, 623, 265, 313, 318, 479, 541, 81, 10205, 65, 6557, 75, 8461, 13);

        testEncoder("The GPT family of models process text using tokens, which are common sequences of characters "
                        + "found in text. The models understand the statistical relationships between these tokens, and excel "
                        + "at producing the next token in a sequence of tokens.",
                464, 402, 11571, 1641, 286, 4981, 1429, 2420, 1262, 16326, 11, 543, 389, 2219, 16311, 286, 3435, 1043,
                287, 2420, 13, 383, 4981, 1833, 262, 13905, 6958, 1022, 777, 16326, 11, 290, 27336, 379, 9194, 262, 1306,
                11241, 287, 257, 8379, 286, 16326, 13);

        testEncoder("{Weird characters <|[!]|>}", 90, 1135, 1447, 3435, 1279, 91, 58, 36463, 91, 29, 92);
    }

    @Test
    public void decoderTest()
    {
        testDecoder("Test", 14402);
        testDecoder("Test this sentence.", 14402, 428, 6827, 13);
        testDecoder("I would like 123 to try this!", 40, 561, 588, 17031, 284, 1949, 428, 0);
        testDecoder("I would like 123 to TRY this!", 40, 561, 588, 17031, 284, 7579, 56, 428, 0);

        testDecoder("é", 2634);

        testDecoder("ő", 129, 239);

        testDecoder("Szeretnék egy ékezetekkel rendelkező magyar mondatot is kipróbálni.",
                50, 89, 31229, 77, 2634, 74, 304, 1360, 38251, 365, 89, 316, 988, 7750, 9851, 417, 365, 89, 129,
                239, 2153, 88, 283, 285, 623, 265, 313, 318, 479, 541, 81, 10205, 65, 6557, 75, 8461, 13);

        testDecoder("The GPT family of models process text using tokens, which are common sequences of characters "
                        + "found in text. The models understand the statistical relationships between these tokens, and excel "
                        + "at producing the next token in a sequence of tokens.",
                464, 402, 11571, 1641, 286, 4981, 1429, 2420, 1262, 16326, 11, 543, 389, 2219, 16311, 286, 3435, 1043,
                287, 2420, 13, 383, 4981, 1833, 262, 13905, 6958, 1022, 777, 16326, 11, 290, 27336, 379, 9194, 262, 1306,
                11241, 287, 257, 8379, 286, 16326, 13);

        testDecoder("{Weird characters <|[!]|>}", 90, 1135, 1447, 3435, 1279, 91, 58, 36463, 91, 29, 92);
    }

    private static Tokenizer getTokenizer()
    {
        var path = new File("src/test/resources").getAbsolutePath() + "/tokenizerConfig/GPT 2";
        var tokenizerConfig = TokenizerConfig.read(path, path);

        return new GPT2Tokenizer(tokenizerConfig);
    }

    private void testEncoder(String text, Integer... tokens)
    {
        var tokenList = Arrays.asList(tokens);
        assertEquals(tokenList, tokenizer.encode(text));
    }

    private void testDecoder(String text, Integer... tokens)
    {
        var tokenList = Arrays.asList(tokens);
        assertEquals(text, tokenizer.decode(tokenList));
    }
}