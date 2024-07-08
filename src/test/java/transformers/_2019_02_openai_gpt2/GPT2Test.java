package transformers._2019_02_openai_gpt2;

import transformers.BaseTransformerTest;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GPT2Test extends BaseTransformerTest
{
    private static final String PATH = "transformers/_2019_02_openai_gpt2";

    @Test
    public void testTransformer()
    {
        var transformer = getTestTransformer(PATH);

        // First run (no previously stored tokens)
        var result1 = transformer.processTokenMain(0, 0, false);

        var expected = new float[] {
                -0.13074648f, -0.22554931f, -0.044833f, -0.0425819f, 0.028920006f, -0.37259027f,
                -0.54749346f, -0.37041944f, 0.25379866f, -0.09713001f, -0.4580743f, -0.08104485f};

        assertVectorEquals(expected, result1, 1e-6f);

        // Second run
        var result2 = transformer.processTokenMain(1, 1, false);

        expected = new float[] {
                0.24844551f, 0.30260512f, -0.1813934f, -0.48943257f, -0.06591987f, -0.6019753f,
                -0.3121546f, 0.008515447f, 0.25019825f, 0.26000157f, -0.3260484f, 0.11481144f};

        assertVectorEquals(expected, result2, 1e-6f);
    }

    @Test
    public void testPreProcess()
    {
        var transformer = getTestTransformer(PATH);

        var output = transformer.preProcessToken(0, 0);

        var expected = new float[] {
                -0.12892373f, -0.23668532f, 0.037134234f, 0.14517331f, 0.015348423f, -0.18393096f,
                -0.20283712f, -0.25750345f, -0.023856042f, -0.1638581f, -0.18360135f, -0.053650603f};

        assertVectorEquals(expected, output, 1e-6f);
    }

    @Test
    public void testAttention()
    {
        var attentionLayer = getTestAttentionLayer(PATH);

        var input = getTestVector(
                -0.12892373f, -0.23668532f, 0.037134234f, 0.14517331f, 0.015348423f, -0.18393096f,
                -0.20283712f, -0.25750345f, -0.023856042f, -0.1638581f, -0.18360135f, -0.053650603f);

        var output = attentionLayer.process(input, false);

        var expected = new float[] {
                -0.029779032f, -0.16865195f, -0.056095753f, -0.08788496f, 0.05123383f, -0.11870614f,
                -0.5170903f, -0.4647233f, 0.113209665f, -0.094052106f, -0.45093378f, -0.011526212f};

        assertVectorEquals(expected, output, 1e-6f);
    }

    @Test
    public void testNeuralNet()
    {
        var neuralNetLayer = getTestNeuralNetLayer(PATH);

        var input = getTestVector(
                -0.029779032f, -0.16865195f, -0.056095753f, -0.08788496f, 0.05123383f, -0.11870614f,
                -0.5170903f, -0.4647233f, 0.113209665f, -0.094052106f, -0.45093378f, -0.011526212f);

        var output = neuralNetLayer.process(input);

        var expected = new float[] {
                -0.13074648f, -0.22554931f, -0.044833f, -0.0425819f, 0.028920006f, -0.37259027f,
                -0.54749346f, -0.37041944f, 0.25379866f, -0.09713001f, -0.4580743f, -0.08104485f};

        assertVectorEquals(expected, output, 1e-6f);
    }

    @Test
    public void testGenerateToken()
    {
        var transformer = getTestTransformer(PATH);

        var input = getTestVector(-0.13074648f, -0.22554931f, -0.044833f, -0.0425819f, 0.028920006f, -0.37259027f,
                -0.54749346f, -0.37041944f, 0.25379866f, -0.09713001f, -0.4580743f, -0.08104485f);

        var output = transformer.generateToken(input, 1);

        // TODO: capture selectBestToken call

        assertEquals(0, output);
    }
}
