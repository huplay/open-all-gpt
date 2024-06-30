package transformer;

import app.IdentifiedException;
import transformer._2018_01_google_transformer.GoogleTransformer;
import transformer._2018_01_google_transformer.GoogleTransformerAttentionLayer;
import transformer._2018_01_google_transformer.GoogleTransformerNeuralNetLayer;
import transformer._2018_06_openai_gpt1.GPT1;
import transformer._2018_06_openai_gpt1.GPT1AttentionLayer;
import transformer._2018_06_openai_gpt1.GPT1NeuralNetLayer;
import transformer._2018_09_facebook_fairseq.Fairseq;
import transformer._2018_09_facebook_fairseq.FairseqAttentionLayer;
import transformer._2018_09_facebook_fairseq.FairseqNeuralNetLayer;
import transformer._2019_02_openai_gpt2.GPT2;
import transformer._2019_02_openai_gpt2.GPT2AttentionLayer;
import transformer._2019_02_openai_gpt2.GPT2NeuralNetLayer;
import transformer._2020_05_openai_gpt3.GPT3;
import transformer._2020_05_openai_gpt3.GPT3AttentionLayer;
import transformer._2020_05_openai_gpt3.GPT3NeuralNetLayer;
import transformer._2021_03_eleutherai_gptneo.GPTNeo;
import transformer._2021_03_eleutherai_gptneo.GPTNeoAttentionLayer;
import transformer._2021_03_eleutherai_gptneo.GPTNeoNeuralNetLayer;
import transformer._2021_06_eleutherai_gptj.GPTJ;
import transformer._2021_06_eleutherai_gptj.GPTJAttentionLayer;
import transformer._2021_06_eleutherai_gptj.GPTJNeuralNetLayer;
import transformer._2022_02_eleutherai_gptneox.GPTNeoX;
import transformer._2022_02_eleutherai_gptneox.GPTNeoXAttentionLayer;
import transformer._2022_02_eleutherai_gptneox.GPTNeoXNeuralNetLayer;
import transformer._2022_05_big_science_bloom.Bloom;
import transformer._2022_05_big_science_bloom.BloomAttentionLayer;
import transformer._2022_05_big_science_bloom.BloomNeuralNetLayer;
import transformer._2022_05_meta_opt.OPT;
import transformer._2022_05_meta_opt.OPTAttentionLayer;
import transformer._2022_05_meta_opt.OPTNeuralNetLayer;
import transformer._2022_05_meta_opt.opt350.OPT350;
import transformer._2022_05_meta_opt.opt350.OPT350AttentionLayer;
import transformer._2022_05_meta_opt.opt350.OPT350NeuralNetLayer;
import transformer._2023_02_meta_llama.Llama;
import transformer._2023_02_meta_llama.LlamaAttentionLayer;
import transformer._2023_02_meta_llama.LlamaNeuralNetLayer;
import transformer._2023_09_mistralai_mistral.Mistral;
import transformer._2023_09_mistralai_mistral.MistralAttentionLayer;
import transformer._2023_09_mistralai_mistral.MistralNeuralNetLayer;
import transformer._2024_02_google_gemma.Gemma;
import transformer._2024_02_google_gemma.GemmaAttentionLayer;
import transformer._2024_02_google_gemma.GemmaNeuralNetLayer;

public enum TransformerType
{
    ORIGINAL_TRANSFORMER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    OPENAI_GPT_3,
    ELEUTHERAI_GPT_NEO,
    ELEUTHERAI_GPT_J,
    ELEUTHERAI_GPT_NEOX,
    BIG_SCIENCE_BLOOM,
    META_FAIRSEQ,
    META_OPT,
    META_OPT_350,
    META_LLAMA,
    GOOGLE_GEMMA,
    MISTRALAI_MISTRAL;

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
            case OPENAI_GPT_3           -> new GPT3();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeo();
            case ELEUTHERAI_GPT_J       -> new GPTJ();
            case ELEUTHERAI_GPT_NEOX    -> new GPTNeoX();
            case BIG_SCIENCE_BLOOM      -> new Bloom();
            case META_FAIRSEQ           -> new Fairseq();
            case META_OPT               -> new OPT();
            case META_OPT_350           -> new OPT350();
            case META_LLAMA             -> new Llama();
            case GOOGLE_GEMMA           -> new Gemma();
            case MISTRALAI_MISTRAL      -> new Mistral();
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
            case OPENAI_GPT_3           -> new GPT3AttentionLayer();
            case BIG_SCIENCE_BLOOM      -> new BloomAttentionLayer();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeoAttentionLayer();
            case ELEUTHERAI_GPT_J       -> new GPTJAttentionLayer();
            case ELEUTHERAI_GPT_NEOX    -> new GPTNeoXAttentionLayer();
            case META_FAIRSEQ           -> new FairseqAttentionLayer();
            case META_OPT               -> new OPTAttentionLayer();
            case META_OPT_350           -> new OPT350AttentionLayer();
            case META_LLAMA             -> new LlamaAttentionLayer();
            case GOOGLE_GEMMA           -> new GemmaAttentionLayer();
            case MISTRALAI_MISTRAL      -> new MistralAttentionLayer();
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
            case OPENAI_GPT_3           -> new GPT3NeuralNetLayer();
            case BIG_SCIENCE_BLOOM      -> new BloomNeuralNetLayer();
            case ELEUTHERAI_GPT_NEO     -> new GPTNeoNeuralNetLayer();
            case ELEUTHERAI_GPT_J       -> new GPTJNeuralNetLayer();
            case ELEUTHERAI_GPT_NEOX    -> new GPTNeoXNeuralNetLayer();
            case META_FAIRSEQ           -> new FairseqNeuralNetLayer();
            case META_OPT               -> new OPTNeuralNetLayer();
            case META_OPT_350           -> new OPT350NeuralNetLayer();
            case META_LLAMA             -> new LlamaNeuralNetLayer();
            case GOOGLE_GEMMA           -> new GemmaNeuralNetLayer();
            case MISTRALAI_MISTRAL      -> new MistralNeuralNetLayer();
        };
    }
}
