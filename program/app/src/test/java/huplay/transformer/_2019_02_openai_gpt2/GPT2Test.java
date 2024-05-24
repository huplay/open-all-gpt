package huplay.transformer._2019_02_openai_gpt2;

import huplay.transformer.BaseTransformerTest;

import huplay.transformer.TransformerFlow;
import org.junit.Test;

public class GPT2Test extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        var config = getTestConfig("transformer/_2019_02_openai_gpt2");
        var flow = new TransformerFlow(config, null, new GPT2());

        // First run (no previously stored tokens)
        var result1 = flow.processTokenMain(0, 0, false);

        var expected = new float[] {
                -0.13074648f, -0.22554931f, -0.044833f, -0.0425819f, 0.028920006f, -0.37259027f,
                -0.54749346f, -0.37041944f, 0.25379866f, -0.09713001f, -0.4580743f, -0.08104485f};

        assertVectorEquals(expected, result1, 1e-6f);

        // Second run
        var result2 = flow.processTokenMain(1, 1, false);

        expected = new float[] {
                0.24844551f, 0.30260512f, -0.1813934f, -0.48943257f, -0.06591987f, -0.6019753f,
                -0.3121546f, 0.008515447f, 0.25019825f, 0.26000157f, -0.3260484f, 0.11481144f};

        assertVectorEquals(expected, result2, 1e-6f);
    }
}
