package transformer._2021_06_eleutherai_gptj;

import org.junit.Ignore;
import transformer.BaseTransformerTest;
import org.junit.Test;

@Ignore
public class GPTJTest extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        var config = getTestConfig("transformer/_2021_06_eleutherai_gptj");
        var transformer = new GPTJ();
        transformer.init(config);
        transformer.initDecoders();

        // First run (no previously stored tokens)
        var result1 = transformer.processTokenMain(0, 5, true);

        // Three hidden states are concatenated, that's why we have 36 values
        var expected = new float[] {
                1.9186351f, -0.003613272f, 0.1504642f, 0.8328369f, -0.0705806f, 0.41953865f,
                0.14085521f, -0.9717808f, -0.68151003f, -0.10620655f, -0.64892626f, -1.0854707f};

        /* Input hidden states (not returned if isInputOnly):
                 0.020477973f, -0.0045622224f, -0.0025853128f, 0.0060729035f, -0.0053859963f, 9.010277E-4f,
                -0.0028172373f, -0.017077204f, -0.013011977f, -0.0058582905f, -0.012819434f, -0.01826822f*/

        /* Attention output hidden states (not returned if isInputOnly):
                -0.0011005357f, -0.0014874976f, -0.0023754076f, -3.4112006E-4f, -2.472683E-4f, -8.199279E-4f,
                -1.9606334E-4f, 8.6140947E-4f, -0.0027803495f, -8.97758E-4f, 9.917559E-4f, 3.7177317E-4f};*/

        assertVectorEquals(expected, result1, 1e-6f);

        // Second run
        var result2 = transformer.processTokenMain(1, 6, false);

        expected = new float[] {
                -0.14471646f, -0.08624163f, 0.29478756f, 0.050751913f, 0.4461088f, 0.13972461f,
                -0.33383986f, -0.19817172f, 0.32056975f, 0.10070647f, -0.07757136f, 0.13080521f};

        assertVectorEquals(expected, result2, 1e-6f);
    }
}
