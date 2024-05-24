package huplay.parameters.quantization;

import huplay.IdentifiedException;
import huplay.parameters.ParameterLoader;
import huplay.parameters.quantization.gptq.GPTQParameterLoader;

public enum QuantizationType
{
    LLM_INT_8,
    GPTQ,
    QLORA,
    AWQ,
    HQQ,
    QUIP,
    AQLM,
    SPQR_3;

    private static QuantizationType getQuantizationType(String quantizationType)
    {
        if (quantizationType == null)
        {
            throw new IdentifiedException("Quantization type isn't specified");
        }

        quantizationType = quantizationType.toUpperCase();
        return QuantizationType.valueOf(quantizationType);
    }

    public static ParameterLoader getParameterLoader(String quantizationType)
    {
        var type = getQuantizationType(quantizationType);
        return switch (type)
        {
            case GPTQ       -> new GPTQParameterLoader();
            default
                    -> throw new IdentifiedException("Unsupported quantization type: " + type);
        };
    }
}
