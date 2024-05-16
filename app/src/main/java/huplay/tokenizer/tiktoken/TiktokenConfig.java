package huplay.tokenizer.tiktoken;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TiktokenConfig
{
    @JsonProperty
    private String version;

    @JsonProperty("added_tokens")
    private List<AddedTokenConfig> addedTokensConfig;

    @JsonProperty
    private ModelConfig model;

    @JsonProperty
    private String type;

    @JsonProperty("unk_token")
    private String unknownToken;

    @JsonProperty("vocab")
    private LinkedHashMap<String, Float> vocabulary;

    // Getters
    public String getVersion() {return version;}
    public List<AddedTokenConfig> getAddedTokensConfig() {return addedTokensConfig;}
    public ModelConfig getModel() {return model;}
    public String getType() {return type;}
    public String getUnknownToken() {return unknownToken;}
    public LinkedHashMap<String, Float> getVocabulary() {return vocabulary;}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AddedTokenConfig
    {
        @JsonProperty
        public int id;

        @JsonProperty
        public String content;

        // Getters
        public int getId() {return id;}
        public String getContent() {return content;}
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ModelConfig
    {
        @JsonProperty
        private String type;

        @JsonProperty("vocab")
        private Map<String, Integer> vocabulary;

        @JsonProperty
        private List<String> merges;

        // Getters
        public String getType() {return type;}
        public Map<String, Integer> getVocabulary() {return vocabulary;}
        public List<String> getMerges() {return merges;}
    }
}
