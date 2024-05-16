package huplay.tokenizer;

import org.junit.Test;

public class SentencePieceTokenizerTest
{
    private static final String PATH = "../tokenizerConfig/Llama1-2";
    //private static final Tokenizer TOKENIZER = new SentencePieceTokenizer(PATH, 32000);

    @Test
    public void test()
    {
        encodeTest(" grabbed", 1, 2646, 1327, 287);
        encodeTest("This is a test", 1, 4013, 338, 263, 1243);
        encodeTest("Autó, teniszütő, ítélet, special hungarian characters", 1, 6147, 29980, 29892, 3006,
                16399, 6621, 30048, 29892, 29871, 2468, 29948, 1026, 29892, 4266, 18757, 13956, 4890);
    }

    private void encodeTest(String text, Integer...expected)
    {
        //assertEquals(Arrays.asList(expected), null/*TOKENIZER.encode(text)*/);
    }
}