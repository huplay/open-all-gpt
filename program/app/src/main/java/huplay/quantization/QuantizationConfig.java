package huplay.quantization;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import huplay.config.ParameterType;

import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class QuantizationConfig
{
    private String quantizationType;
    private String variant;
    private List<String> parameters;
    private Map<String, String> naming;
    private String outputFloatType;
    private boolean transposeMatrix;

    // Getters
    public String getQuantizationType() {return quantizationType;}
    public String getVariant() {return variant;}
    public List<String> getParameters() {return parameters;}
    public Map<String, String> getNaming() {return naming;}
    public String getOutputFloatType() {return outputFloatType;}
    public boolean getTransposeMatrix() {return transposeMatrix;}

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
