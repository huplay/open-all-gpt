package huplay.transformer._2021_06_eleuther_gptj;

import huplay.transformer.BaseTransformerTest;
import huplay.transformer.TransformerFlow;
import org.junit.Ignore;
import org.junit.Test;

@Ignore
public class GPTJTest extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        var config = getTestConfig("transformer/_2021_06_eleuther_gptj");
        var flow = new TransformerFlow(config, null, new GPTJ());

        // First run (no previously stored tokens)
        var result1 = flow.processTokenMain(0, 0, false);

        var expected = new float[] {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertVectorEquals(expected, result1, 1e-6f);

        // Second run
        var result2 = flow.processTokenMain(1, 1, false);

        expected = new float[] {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertVectorEquals(expected, result2, 1e-6f);
    }
}
