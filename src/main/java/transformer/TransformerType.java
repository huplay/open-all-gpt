package transformer;

import app.IdentifiedException;
import transformer._2018_01_google_transformer.GoogleTransformer;
import transformer._2018_01_google_transformer.GoogleTransformerAttentionLayer;
import transformer._2018_01_google_transformer.GoogleTransformerNeuralNetLayer;
import transformer._2018_06_openai_gpt1.GPT1AttentionLayer;
import transformer._2018_06_openai_gpt1.GPT1NeuralNetLayer;
import transformer._2019_02_openai_gpt2.GPT2AttentionLayer;
import transformer._2019_02_openai_gpt2.GPT2NeuralNetLayer;
import transformer._2021_03_eleuther_gptneo.GPTNeo;
import transformer._2021_03_eleuther_gptneo.GPTNeoAttentionLayer;
import transformer._2021_03_eleuther_gptneo.GPTNeoNeuralNetLayer;
import transformer._2021_06_eleuther_gptj.GPTJ;
import transformer._2021_06_eleuther_gptj.GPTJAttentionLayer;
import transformer._2021_06_eleuther_gptj.GPTJNeuralNetLayer;
import transformer._2022_05_big_science_bloom.Bloom;
import transformer._2022_05_big_science_bloom.BloomAttentionLayer;
import transformer._2022_05_big_science_bloom.BloomNeuralNetLayer;
import transformer._2023_02_meta_llama.Llama;
import transformer._2018_06_openai_gpt1.GPT1;
import transformer._2019_02_openai_gpt2.GPT2;
import transformer._2023_02_meta_llama.LlamaAttentionLayer;
import transformer._2023_02_meta_llama.LlamaNeuralNetLayer;
import transformer._2024_02_google_gemma.Gemma;
import transformer._2024_02_google_gemma.GemmaAttentionLayer;
import transformer._2024_02_google_gemma.GemmaNeuralNetLayer;

public enum TransformerType
{
    ORIGINAL_TRANSFORMER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    ELEUTHERAI_GPT_NEO,
    ELEUTHERAI_GPT_J,
    BIG_SCIENCE_BLOOM,
    GOOGLE_GEMMA,
    META_LLAMA;

    private static TransformerType getTransformerType(String transformerType)
    {
        if (transformerType == null)
        {
            throw new IdentifiedException("Transformer type isn't specified");
        }

        transformerType = transformerType.toUpperCase();
        return TransformerType.valueOf(transformerType);
    }

    public static BaseTransformer getTransformer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            case ORIGINAL_TRANSFORMER   -> new GoogleTransformer();
            case OPENAI_GPT_1           -> new GPT1();
            case OPENAI_GPT_2           -> new GPT2();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeo();
            case ELEUTHERAI_GPT_J       -> new GPTJ();
            case BIG_SCIENCE_BLOOM      -> new Bloom();
            case GOOGLE_GEMMA           -> new Gemma();
            case META_LLAMA             -> new Llama();
        };
    }

    public static BaseAttentionLayer getAttentionLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            case ORIGINAL_TRANSFORMER   -> new GoogleTransformerAttentionLayer();
            case OPENAI_GPT_1           -> new GPT1AttentionLayer();
            case OPENAI_GPT_2           -> new GPT2AttentionLayer();
            case BIG_SCIENCE_BLOOM      -> new BloomAttentionLayer();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeoAttentionLayer();
            case ELEUTHERAI_GPT_J       -> new GPTJAttentionLayer();
            case GOOGLE_GEMMA           -> new GemmaAttentionLayer();
            case META_LLAMA             -> new LlamaAttentionLayer();
        };
    }

    public static BaseNeuralNetLayer getNeuralNetLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            case ORIGINAL_TRANSFORMER   -> new GoogleTransformerNeuralNetLayer();
            case OPENAI_GPT_1           -> new GPT1NeuralNetLayer();
            case OPENAI_GPT_2           -> new GPT2NeuralNetLayer();
            case BIG_SCIENCE_BLOOM      -> new BloomNeuralNetLayer();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeoNeuralNetLayer();
            case ELEUTHERAI_GPT_J       -> new GPTJNeuralNetLayer();
            case GOOGLE_GEMMA           -> new GemmaNeuralNetLayer();
            case META_LLAMA             -> new LlamaNeuralNetLayer();
        };
    }
}
