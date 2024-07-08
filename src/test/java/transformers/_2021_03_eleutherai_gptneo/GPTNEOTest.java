package transformers._2021_03_eleutherai_gptneo;

import transformers.BaseTransformerTest;
import org.junit.Test;

public class GPTNEOTest extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        var config = getTestConfig("transformers/_2021_03_eleutherai_gptneo");
        var transformer = new GPTNeo();
        transformer.init(config);
        transformer.initDecoders();

        // First run (no previously stored tokens)
        var result1 = transformer.processTokenMain(0, 0, false);

        var expected = new float[] {
                0.5569178f, -0.866098f, 0.2154375f, -0.3094095f, -0.8206209f, 0.23917273f,
                0.42553633f, -1.3366764f, 0.7802603f, 0.5322888f, 0.046592668f, -0.07823685f};

        assertVectorEquals(expected, result1, 1e-6f);

        // Second run
        var result2 = transformer.processTokenMain(1, 1, false);

        expected = new float[] {
                0.2945429f, 1.1000853f, -0.426582f, 0.013817161f, 0.46769962f, -0.34113365f,
                0.35800916f, -0.9641304f, 0.8715731f, 0.030839011f, 0.18299714f, -0.19659953f};

        assertVectorEquals(expected, result2, 1e-6f);
    }
}
