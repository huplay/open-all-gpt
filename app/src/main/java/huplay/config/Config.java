package huplay.config;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.file.SafetensorsReader;

import java.io.IOException;
import java.util.*;

/**
 * Holder of the config file describing the details of the model.
 * This file usually provided by the model, so it can be downloaded from the repo
 * For the models it is missing, or has a very different structure, naming,
 * a manageable config.json is added to the modelConfig folder
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class Config
{
    private Arguments arguments;
    private ModelConfig modelConfig;
    private TokenizerConfig tokenizerConfig;
    private SafetensorsReader reader;
    private boolean isCalculationOnly;

    @JsonAlias({"n_embd", "hidden_size", "n_embed"})
    private int hiddenSize;

    @JsonAlias({"intermediate_size", "n_inner"})
    private Integer feedForwardSize;

    @JsonAlias({"n_layer", "num_hidden_layers", "num_layers"})
    private int decoderCount;

    @JsonAlias({"n_head", "num_attention_heads", "num_heads"})
    private int headCount;

    @JsonAlias({"layer_norm_epsilon", "rms_norm_eps"})
    private float epsilon;

    @JsonAlias({"n_ctx", "max_position_embeddings", "n_positions"})
    private int contextSize;

    @JsonAlias({"vocab_size"})
    private int tokenCount;

    @JsonAlias({"eos_token_id"})
    private int endOfTextToken;

    private Map<String, Object> allEntries;

    public static Config readConfig(Arguments arguments, ModelConfig modelConfig, TokenizerConfig tokenizerConfig,
                                    SafetensorsReader reader)
    {
        var configFile = modelConfig.findFile("config.json");

        if (!configFile.exists())
        {
            throw new IdentifiedException("Can't find the config file. (" + configFile.getName() + ")");
        }

        try
        {
            var objectMapper = new ObjectMapper();
            var config = objectMapper.readValue(configFile, Config.class);

            config.arguments = arguments;
            config.modelConfig = modelConfig;
            config.tokenizerConfig = tokenizerConfig;
            config.reader = reader;

            if (arguments != null)
            {
                config.isCalculationOnly = arguments.isCalculationOnly();
            }

            // If the feed forward size isn't configured set it as 4 times of the hidden size
            if (config.feedForwardSize == null) config.feedForwardSize = 4 * config.hiddenSize;

            // At some models the context size is unlimited, so this value isn't configured. Set it to max possible.
            if (config.contextSize == 0) config.contextSize = Integer.MAX_VALUE;

            var typeRef = new TypeReference<Map<String, Object>>() {};
            config.allEntries = objectMapper.readValue(configFile, typeRef);

            return config;
        }
        catch (IOException e)
        {
            System.out.println(e.getMessage());
            throw new IdentifiedException("Can't read the config file.\n" + configFile.getName(), e);
        }
    }

    public int getInt(String key)
    {
        try
        {
            var value = allEntries.get(key);
            if (value == null)
            {
                throw new IdentifiedException("Integer configuration is missing: " + key);
            }

            return toInt(value);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Cannot read integer property: " + key + " exception: " + e.getMessage());
        }
    }

    private int toInt(Object value)
    {
        try
        {
            return Integer.parseInt(value.toString());
        }
        catch (Exception e)
        {
            throw new IdentifiedException("The provided properties value can't be converted to integer (" + value + ").");
        }
    }

    public int getIntOptional(String key, int defaultValue)
    {
        try
        {
            var value = allEntries.get(key);
            return value == null ? defaultValue : toInt(value);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Cannot read integer property: " + key + " exception: " + e.getMessage());
        }
    }

    // Getters
    public ModelConfig getModelConfig() {return modelConfig;}
    public SafetensorsReader getReader() {return reader;}
    public int getHiddenSize() {return hiddenSize;}
    public int getFeedForwardSize() {return feedForwardSize;}
    public int getDecoderCount() {return decoderCount;}
    public int getHeadCount() {return headCount;}
    public float getEpsilon() {return epsilon;}
    public int getContextSize() {return contextSize;}
    public int getTokenCount() {return tokenCount;}
    public int getEndOfTextToken() {return endOfTextToken;}

    // Getters, setters to Arguments
    public String getConfigRoot() {return arguments.getConfigRoot();}
    public String getDownloadRoot() {return arguments.getDownloadRoot();}
    public String getConfigPath() {return arguments.getConfigPath();}
    public String getModelPath() {return arguments.getModelPath();}
    public int getLengthLimit() {return arguments.getLengthLimit();}
    public int getTopK() {return arguments.getTopK();}
    public boolean isCalculationOnly() {return isCalculationOnly;}
    public void setCalculationOnly(boolean isCalculationOnly) {this.isCalculationOnly = isCalculationOnly;}
    public Integer getRequestedMemorySize() {return arguments.getRequestedMemorySize();}

    // Getters to ModelConfig
    public String getName() {return modelConfig.getName();}
    public String getTransformerType() {return modelConfig.getTransformerType();}
    public String getTokenizerType() {return tokenizerConfig.getTokenizerType();}
    public String getRepo() {return modelConfig.getRepo();}
    public String getBranch() {return modelConfig.getBranch();}
    public List<String> getFiles() {return modelConfig.getFiles();}
    public Map<String, String> getFileNameOverrides() {return modelConfig.getFileNameOverrides();}
    public String getParameterNaming() {return modelConfig.getParameterNaming();}
    public String getDecoderNameFormat() {return modelConfig.getDecoderParameterNaming();}
    public Map<String, String> getParameterNameOverrides() {return modelConfig.getParameterNameOverrides();}
    public Integer getMemorySize() {return modelConfig.getMemorySize();}
    public Map<BlockType, Integer> getMemorySizes() {return modelConfig.getMemorySizes();}

    public int getHeadSize()
    {
        return hiddenSize / headCount;
    }
}
