package quantization;

import app.IdentifiedException;
import config.Config;
import quantization.gptq.GptqQuantizer;
import quantization.llmInt8.LlmInt8Quantizer;
import quantization.qlora.QloraQuantizer;

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

    public static AbstractQuantizer getQuantizer(Config config, String quantizationType)
    {
        var type = getQuantizationType(quantizationType);
        return switch (type)
        {
            case LLM_INT_8  -> new LlmInt8Quantizer(config);
            case QLORA      -> new QloraQuantizer(config);
            case GPTQ       -> new GptqQuantizer(config);
        };
    }
}
