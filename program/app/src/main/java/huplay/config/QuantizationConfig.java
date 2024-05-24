package huplay.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class QuantizationConfig
{
    private String quantizationType;
    private List<String> parameters;
    private Map<String, String> naming;

    public String getQuantizationType()
    {
        return quantizationType;
    }

    public List<String> getParameters()
    {
        return parameters;
    }

    public Map<String, String> getNaming()
    {
        return naming;
    }
}
