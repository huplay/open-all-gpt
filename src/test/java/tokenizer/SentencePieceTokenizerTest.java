package tokenizer;

import config.TokenizerConfig;
import tokenizer.Tokenizer;
import tokenizer.sentencePiece.SentencePieceTokenizer;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class SentencePieceTokenizerTest
{
    private static final Tokenizer tokenizer = getTokenizer();

    @Test
    public void test()
    {
        encodeTest("This is a test", 1, 19988, 2236, 2236, 27137, 893);
        encodeTest("Autó, teniszütő, ítélet, special hungarian characters", 1, 23184, 12985, 698, 15755,
                29993, 698, 12985, 29983, 698, 645, 698, 462, 21478, 15755, 617, 27137, 3905, 29879);

        //encodeTest(" grabbed", 1, 2646, 1327, 287);
        //encodeTest("This is a test", 1, 4013, 338, 263, 1243);
        //encodeTest("Autó, teniszütő, ítélet, special hungarian characters", 1, 6147, 29980, 29892, 3006,
        //        16399, 6621, 30048, 29892, 29871, 2468, 29948, 1026, 29892, 4266, 18757, 13956, 4890);
    }

    private void encodeTest(String text, Integer...expected)
    {
        assertEquals(Arrays.asList(expected), tokenizer.encode(text));
    }

    private static Tokenizer getTokenizer()
    {
        var path = new File("src/test/resources").getAbsolutePath() + "/tokenizerConfig/Llama";
        var tokenizerConfig = TokenizerConfig.read(path, path);

        return new SentencePieceTokenizer(tokenizerConfig, "TINY");
    }
}