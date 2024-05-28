package huplay.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.quantization.QuantizationConfig;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Holder of the configuration stored in the model.json file
 * This is the file to describe where is the model, how to download, which file are needed, what is the config file name
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig implements RepoConfig
{
    private String configPath;
    private String downloadPath;

    private String name;
    private String transformerType;
    private String repo;
    private String branch;
    private List<String> files;
    private Map<String, String> fileNameOverrides;
    private String parameterNaming;
    private String decoderParameterNaming;
    private Map<String, String> parameterNameOverrides;
    private Integer memorySize;
    private Map<BlockType, Integer> memorySizes;
    private Config configOverride;
    private QuantizationConfig quantization;

    public static ModelConfig read(String configPath, String downloadPath)
    {
        var modelConfigJson = configPath + "/model.json";

        var modelConfigFile = new File(modelConfigJson);
        if (!modelConfigFile.exists())
        {
            throw new IdentifiedException("Model config file is missing (" + modelConfigJson + ")");
        }

        try
        {
            var modelConfig = new ObjectMapper().readValue(modelConfigFile, ModelConfig.class);
            modelConfig.init(configPath, downloadPath);

            return modelConfig;
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot read model.json (" + modelConfigJson + ")");
        }
    }

    private void init(String configPath, String downloadPath)
    {
        this.configPath = configPath;
        this.downloadPath = downloadPath;

        if (fileNameOverrides == null) fileNameOverrides = new HashMap<>();

        if (parameterNaming == null) parameterNaming = "{name}";

        if (decoderParameterNaming == null) decoderParameterNaming = "{decoderId}.{name}";

        if (parameterNameOverrides == null) parameterNameOverrides = new HashMap<>();
    }

    // Getters
    public String getDownloadPath() {return downloadPath;}
    public String getName() {return name;}
    public String getTransformerType() {return transformerType;}
    public String getRepo() {return repo;}
    public String getBranch() {return branch;}
    public List<String> getFiles() {return files;}
    public Map<String, String> getFileNameOverrides() {return fileNameOverrides;}
    public String getParameterNaming() {return parameterNaming;}
    public String getDecoderParameterNaming() {return decoderParameterNaming;}
    public Map<String, String> getParameterNameOverrides() {return parameterNameOverrides;}
    public Integer getMemorySize() {return memorySize;}
    public Map<BlockType, Integer> getMemorySizes() {return memorySizes;}
    public Config getConfigOverride() {return configOverride;}
    public QuantizationConfig getQuantizationConfig() {return quantization;}

    public String resolveFileName(String name)
    {
        var overriddenName = fileNameOverrides.get(name);
        return overriddenName == null ? name : overriddenName;
    }

    public File findFile(String name)
    {
        var resolvedName = resolveFileName(name);
        var file = new File(downloadPath + "/" + resolvedName);
        if (file.exists() && file.isFile())
        {
            return file;
        }
        else
        {
            return new File(configPath + "/" + resolvedName);
        }
    }
}
