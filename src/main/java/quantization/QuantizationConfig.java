package quantization;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import config.ParameterType;
import math.dataType.DataType;

import java.util.List;
import java.util.Map;

import static ui.TextUtil.equalsIgnoreTypo;

@JsonIgnoreProperties(ignoreUnknown = true)
public class QuantizationConfig
{
    private String quantizationType;
    private List<String> parameters;
    private Map<String, String> naming;
    private String outputFloatType;
    private boolean deQuantizeOnLoad;

    // Getters
    public String getQuantizationType() {return quantizationType;}
    public List<String> getParameters() {return parameters;}
    public Map<String, String> getNaming() {return naming;}
    public boolean getDeQuantizeOnLoad() {return deQuantizeOnLoad;}

    public DataType getOutputFloatType()
    {
        if (outputFloatType == null || equalsIgnoreTypo("FLOAT32", outputFloatType))
        {
            return DataType.FLOAT_32;
        }
        else if (equalsIgnoreTypo("FLOAT16", outputFloatType))
        {
            return DataType.FLOAT_16;
        }
        else if (equalsIgnoreTypo("BFLOAT16", outputFloatType) || equalsIgnoreTypo("BRAINFLOAT16", outputFloatType))
        {
            return DataType.BRAIN_FLOAT_16;
        }

        throw new RuntimeException("Unsupported output float type: " + outputFloatType);
    }

    public boolean isQuantized(ParameterType parameterType, String id)
    {
        if (parameters == null)
        {
            return parameterType.isWeight();
        }
        else
        {
            return parameters.contains(id);
        }
    }
}
