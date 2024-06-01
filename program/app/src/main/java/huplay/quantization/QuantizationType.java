package huplay.quantization;

import huplay.IdentifiedException;
import huplay.config.Config;
import huplay.parameters.ParameterLoader;
import huplay.quantization.gptq.GptqParameterLoader;
import huplay.quantization.llmInt8.LlmInt8ParameterLoader;
import huplay.quantization.qlora.QloraParameterLoader;

public enum QuantizationType
{
    LLM_INT_8,
    QLORA,
    GPTQ;
    //AWQ,
    //HQQ,
    //QUIP,
    //AQLM,
    //SPQR_3;

    private static QuantizationType getQuantizationType(String quantizationType)
    {
        if (quantizationType == null)
        {
            throw new IdentifiedException("Quantization type isn't specified");
        }

        quantizationType = quantizationType.toUpperCase();
        return QuantizationType.valueOf(quantizationType);
    }

    public static ParameterLoader getParameterLoader(Config config)
    {
        var type = getQuantizationType(config.getQuantizationConfig().getQuantizationType());
        return switch (type)
        {
            case LLM_INT_8  -> new LlmInt8ParameterLoader(config);
            case QLORA      -> new QloraParameterLoader(config);
            case GPTQ       -> new GptqParameterLoader(config);
        };
    }
}
