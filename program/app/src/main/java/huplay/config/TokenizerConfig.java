package huplay.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;

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
public class TokenizerConfig implements RepoConfig
{
    private String configPath;
    private String downloadPath;

    private String tokenizerType;
    private String repo;
    private String branch;
    private List<String> files;
    private Map<String, String> fileNameOverrides;

    public static TokenizerConfig read(String configPath, String downloadPath)
    {
        var tokenizerConfigJson = configPath + "/tokenizer.json";

        var tokenizerConfigFile = new File(tokenizerConfigJson);
        if (!tokenizerConfigFile.exists())
        {
            throw new IdentifiedException("Tokenizer config file is missing (" + tokenizerConfigJson + ")");
        }

        try
        {
            var tokenizerConfig = new ObjectMapper().readValue(tokenizerConfigFile, TokenizerConfig.class);
            tokenizerConfig.init(configPath, downloadPath);

            return tokenizerConfig;
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot read model.json (" + tokenizerConfigJson + ")");
        }
    }

    private void init(String configPath, String downloadPath)
    {
        this.configPath = configPath;
        this.downloadPath = downloadPath;

        if (fileNameOverrides == null) fileNameOverrides = new HashMap<>();
    }

    // Getters
    public String getTokenizerType() {return tokenizerType;}
    public String getRepo() {return repo;}
    public String getBranch() {return branch;}
    public List<String> getFiles() {return files;}
    public Map<String, String> getFileNameOverrides() {return fileNameOverrides;}

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
