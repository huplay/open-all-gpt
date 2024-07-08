package transformer;

import app.IdentifiedException;
import transformer.parallel.ParallelBaseAttentionLayer;
import transformer.parallel.ParallelBaseNeuralNetLayer;
import transformer.parallel.ParallelBaseTransformer;
import transformer.serial.BaseAttentionLayer;
import transformer.serial.BaseNeuralNetLayer;
import transformer.serial.BaseTransformer;
import transformers._2018_01_google_transformer.GoogleTransformer;
import transformers._2018_01_google_transformer.GoogleTransformerAttentionLayer;
import transformers._2018_01_google_transformer.GoogleTransformerNeuralNetLayer;
import transformers._2018_06_openai_gpt1.GPT1;
import transformers._2018_06_openai_gpt1.GPT1AttentionLayer;
import transformers._2018_06_openai_gpt1.GPT1NeuralNetLayer;
import transformers._2018_09_facebook_fairseq.Fairseq;
import transformers._2018_09_facebook_fairseq.FairseqAttentionLayer;
import transformers._2018_09_facebook_fairseq.FairseqNeuralNetLayer;
import transformers._2019_02_openai_gpt2.parallel.GPT2;
import transformers._2019_02_openai_gpt2.parallel.GPT2AttentionLayer;
import transformers._2019_02_openai_gpt2.parallel.GPT2NeuralNetLayer;
import transformers._2020_05_openai_gpt3.GPT3;
import transformers._2020_05_openai_gpt3.GPT3AttentionLayer;
import transformers._2020_05_openai_gpt3.GPT3NeuralNetLayer;
import transformers._2021_03_eleutherai_gptneo.GPTNeo;
import transformers._2021_03_eleutherai_gptneo.GPTNeoAttentionLayer;
import transformers._2021_03_eleutherai_gptneo.GPTNeoNeuralNetLayer;
import transformers._2021_06_eleutherai_gptj.GPTJ;
import transformers._2021_06_eleutherai_gptj.GPTJAttentionLayer;
import transformers._2021_06_eleutherai_gptj.GPTJNeuralNetLayer;
import transformers._2022_02_eleutherai_gptneox.GPTNeoX;
import transformers._2022_02_eleutherai_gptneox.GPTNeoXAttentionLayer;
import transformers._2022_02_eleutherai_gptneox.GPTNeoXNeuralNetLayer;
import transformers._2022_05_big_science_bloom.Bloom;
import transformers._2022_05_big_science_bloom.BloomAttentionLayer;
import transformers._2022_05_big_science_bloom.BloomNeuralNetLayer;
import transformers._2022_05_meta_opt.OPT;
import transformers._2022_05_meta_opt.OPTAttentionLayer;
import transformers._2022_05_meta_opt.OPTNeuralNetLayer;
import transformers._2022_05_meta_opt.opt350.OPT350;
import transformers._2022_05_meta_opt.opt350.OPT350AttentionLayer;
import transformers._2022_05_meta_opt.opt350.OPT350NeuralNetLayer;
import transformers._2023_02_meta_llama.Llama;
import transformers._2023_02_meta_llama.LlamaAttentionLayer;
import transformers._2023_02_meta_llama.LlamaNeuralNetLayer;
import transformers._2023_09_mistralai_mistral.Mistral;
import transformers._2023_09_mistralai_mistral.MistralAttentionLayer;
import transformers._2023_09_mistralai_mistral.MistralNeuralNetLayer;
import transformers._2024_02_google_gemma.Gemma;
import transformers._2024_02_google_gemma.GemmaAttentionLayer;
import transformers._2024_02_google_gemma.GemmaNeuralNetLayer;
import transformers._2024_06_google_gemma2.Gemma2;
import transformers._2024_06_google_gemma2.Gemma2AttentionLayer;
import transformers._2024_06_google_gemma2.Gemma2NeuralNetLayer;

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
    GOOGLE_GEMMA_2,
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
            case OPENAI_GPT_2           -> new transformers._2019_02_openai_gpt2.serial.GPT2();
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
            case GOOGLE_GEMMA_2         -> new Gemma2();
            case MISTRALAI_MISTRAL      -> new Mistral();
        };
    }

    public static ParallelBaseTransformer getParallelTransformer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            //case ORIGINAL_TRANSFORMER   -> new ParallelGoogleTransformer();
            //case OPENAI_GPT_1           -> new ParallelGPT1();
            case OPENAI_GPT_2           -> new GPT2();
            //case OPENAI_GPT_3           -> new ParallelGPT3();
            //case ELEUTHERAI_GPT_NEO     -> new ParallelGPTNeo();
            //case ELEUTHERAI_GPT_J       -> new ParallelGPTJ();
            //case ELEUTHERAI_GPT_NEOX    -> new ParallelGPTNeoX();
            //case BIG_SCIENCE_BLOOM      -> new ParallelBloom();
            //case META_FAIRSEQ           -> new ParallelFairseq();
            //case META_OPT               -> new ParallelOPT();
            //case META_OPT_350           -> new ParallelOPT350();
            //case META_LLAMA             -> new ParallelLlama();
            //case GOOGLE_GEMMA           -> new ParallelGemma();
            //case GOOGLE_GEMMA_2         -> new ParallelGemma2();
            //case MISTRALAI_MISTRAL      -> new ParallelMistral();
            default ->
                    throw new IdentifiedException("Unsupported parallel transformer type" + transformerType);
        };
    }

    public static BaseAttentionLayer getAttentionLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            case ORIGINAL_TRANSFORMER   -> new GoogleTransformerAttentionLayer();
            case OPENAI_GPT_1           -> new GPT1AttentionLayer();
            case OPENAI_GPT_2           -> new transformers._2019_02_openai_gpt2.serial.GPT2AttentionLayer();
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
            case GOOGLE_GEMMA_2         -> new Gemma2AttentionLayer();
            case MISTRALAI_MISTRAL      -> new MistralAttentionLayer();
        };
    }

    public static ParallelBaseAttentionLayer getParallelAttentionLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            //case ORIGINAL_TRANSFORMER   -> new ParallelGoogleTransformerAttentionLayer();
            //case OPENAI_GPT_1           -> new ParallelGPT1AttentionLayer();
            case OPENAI_GPT_2             -> new GPT2AttentionLayer();
            //case OPENAI_GPT_3           -> new ParallelGPT3AttentionLayer();
            //case BIG_SCIENCE_BLOOM      -> new ParallelBloomAttentionLayer();
            //case ELEUTHERAI_GPT_NEO     -> new ParallelGPTNeoAttentionLayer();
            //case ELEUTHERAI_GPT_J       -> new ParallelGPTJAttentionLayer();
            //case ELEUTHERAI_GPT_NEOX    -> new ParallelGPTNeoXAttentionLayer();
            //case META_FAIRSEQ           -> new ParallelFairseqAttentionLayer();
            //case META_OPT               -> new ParallelOPTAttentionLayer();
            //case META_OPT_350           -> new ParallelOPT350AttentionLayer();
            //case META_LLAMA             -> new ParallelLlamaAttentionLayer();
            //case GOOGLE_GEMMA           -> new ParallelGemmaAttentionLayer();
            //case GOOGLE_GEMMA_2         -> new ParallelGemma2AttentionLayer();
            //case MISTRALAI_MISTRAL      -> new ParallelMistralAttentionLayer();
            default ->
                    throw new IdentifiedException("Unsupported parallel transformer type" + transformerType);
        };
    }

    public static BaseNeuralNetLayer getNeuralNetLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            case ORIGINAL_TRANSFORMER   -> new GoogleTransformerNeuralNetLayer();
            case OPENAI_GPT_1           -> new GPT1NeuralNetLayer();
            case OPENAI_GPT_2           -> new transformers._2019_02_openai_gpt2.serial.GPT2NeuralNetLayer();
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
            case GOOGLE_GEMMA_2         -> new Gemma2NeuralNetLayer();
            case MISTRALAI_MISTRAL      -> new MistralNeuralNetLayer();
        };
    }

    public static ParallelBaseNeuralNetLayer getParallelNeuralNetLayer(String transformerType)
    {
        var type = getTransformerType(transformerType);
        return switch (type)
        {
            //case ORIGINAL_TRANSFORMER   -> new ParallelGoogleTransformerNeuralNetLayer();
            //case OPENAI_GPT_1           -> new ParallelGPT1NeuralNetLayer();
            case OPENAI_GPT_2           -> new GPT2NeuralNetLayer();
            //case OPENAI_GPT_3           -> new ParallelGPT3NeuralNetLayer();
            //case BIG_SCIENCE_BLOOM      -> new ParallelBloomNeuralNetLayer();
            //case ELEUTHERAI_GPT_NEO     -> new ParallelGPTNeoNeuralNetLayer();
            //case ELEUTHERAI_GPT_J       -> new ParallelGPTJNeuralNetLayer();
            //case ELEUTHERAI_GPT_NEOX    -> new ParallelGPTNeoXNeuralNetLayer();
            //case META_FAIRSEQ           -> new ParallelFairseqNeuralNetLayer();
            //case META_OPT               -> new ParallelOPTNeuralNetLayer();
            //case META_OPT_350           -> new ParallelOPT350NeuralNetLayer();
            //case META_LLAMA             -> new ParallelLlamaNeuralNetLayer();
            //case GOOGLE_GEMMA           -> new ParallelGemmaNeuralNetLayer();
            //case GOOGLE_GEMMA_2         -> new ParallelGemma2NeuralNetLayer();
            //case MISTRALAI_MISTRAL      -> new ParallelMistralNeuralNetLayer();
            default ->
                    throw new IdentifiedException("Unsupported parallel transformer type" + transformerType);
        };
    }
}
