package huplay.parameters.safetensors;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SafetensorsModel
{
    @JsonProperty("__metadata__")
    private Map<String, String> metadata;

    private final Map<String, TensorModel> tensors = new HashMap<>();

    public SafetensorsModel()
    {
    }

    public SafetensorsModel(Map<String, String> metadata)
    {
        this.metadata = metadata;
    }

    // Getters
    public Map<String, String> getMetadata() {return metadata;}

    @JsonAnySetter
    public void addTensor(String parameterId, TensorModel tensor)
    {
        tensors.put(parameterId, tensor);
    }

    @JsonAnyGetter
    public Map<String, TensorModel> getTensors()
    {
        return tensors;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TensorModel
    {
        @JsonProperty("dtype")
        private String dataType;

        @JsonProperty("shape")
        private List<Integer> shape;

        @JsonProperty("data_offsets")
        private List<Long> dataOffsets;

        public TensorModel()
        {
        }

        public TensorModel(String dataType, List<Integer> shape, Long startOffset, long endOffset)
        {
            this.dataType = dataType;
            this.shape = shape;
            this.dataOffsets = new ArrayList<>(2);
            this.dataOffsets.add(startOffset);
            this.dataOffsets.add(endOffset);
        }

        // Getters
        public String getDataType() {return dataType;}
        public List<Integer> getShape() {return shape;}
        public List<Long> getDataOffsets() {return dataOffsets;}
    }
}
