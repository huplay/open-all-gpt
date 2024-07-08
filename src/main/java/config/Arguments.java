package config;

import java.io.*;

import static ui.TextUtil.equalsIgnoreCase;
import static ui.TextUtil.readInt;

/**
 * Holder of the app's input parameters
 */
public class Arguments
{
    private static final String ARG_CALC = "-calc";
    private static final String ARG_MAX = "-max";
    private static final String ARG_TOP_K = "-topK";
    private static final String ARG_MEM = "-mem";
    private static final String ARG_PARALLEL = "-parallel";

    // The root folder of the model configurations
    // The default is the "models", but it can be overridden by the OPEN_ALL_GPT_MODELS_ROOT environment variable
    private final String configRoot;

    // The root folder of the model parameters
    // The default is the "download", but it can be overridden by the OPEN_ALL_GPT_DOWNLOAD_ROOT environment variable
    private final String downloadRoot;

    // The relative path of the selected model
    private String modelId;

    private final int lengthLimit;
    private final int topK;
    private boolean isCalculationOnly;
    private final Integer requestedMemorySize;
    private final boolean isParallel;
    private final String serverAddress;
    private final Integer port;

    public Arguments(String configRoot, String downloadRoot, String modelId,
                     int lengthLimit, int topK, boolean isCalculationOnly, int requestedMemorySize,
                     boolean isParallel, String serverAddress, Integer port)
    {
        this.configRoot = configRoot;
        this.downloadRoot = downloadRoot;
        this.modelId = modelId;
        this.lengthLimit = lengthLimit;
        this.topK = topK;
        this.isCalculationOnly = isCalculationOnly;
        this.requestedMemorySize = requestedMemorySize;
        this.isParallel = isParallel;
        this.serverAddress = serverAddress;
        this.port = port;
    }

    public static Arguments read(String[] args)
    {
        var file = new File("models");
        var configRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_MODELS_ROOT", file.getAbsolutePath());

        file = new File("download");
        var downloadRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_DOWNLOAD_ROOT", file.getAbsolutePath());

        configRoot = configRoot.replace("\\", "/");
        downloadRoot = downloadRoot.replace("\\", "/");

        // Default values
        String modelPath = null;
        var maxLength = 25;
        var topK = 40;
        var requestedMemorySize = 0;
        var isCalculationOnly = false;
        var isParallel = false;
        String serverAddress = null;
        Integer port = null;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
            for (var arg : args)
            {
                if (arg.charAt(0) == '-')
                {
                    if (equalsIgnoreCase(arg, ARG_CALC)) isCalculationOnly = true;
                    if (equalsIgnoreCase(arg, ARG_PARALLEL)) isParallel = true;
                    else
                    {
                        var parts = arg.split("=");
                        if (parts.length == 2)
                        {
                            var key = parts[0];
                            var value = parts[1];

                            if (equalsIgnoreCase(key, ARG_MAX)) maxLength = readInt(value, maxLength);
                            else if (equalsIgnoreCase(key, ARG_TOP_K)) topK = readInt(value, topK);
                            else if (equalsIgnoreCase(key, ARG_MEM)) requestedMemorySize = readInt(value, 0);
                        }
                        else
                        {
                            System.out.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                        }
                    }
                }
                else if (modelPath != null)
                {
                    System.out.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
                else
                {
                    modelPath = removeDoubleQuotes(arg);
                }
            }
        }

        return new Arguments(configRoot, downloadRoot, modelPath, maxLength, topK, isCalculationOnly,
                requestedMemorySize, isParallel, serverAddress, port);
    }

    // Getters, setters
    public String getConfigRoot() {return configRoot;}
    public String getDownloadRoot() {return downloadRoot;}
    public String getModelId() {return modelId;}
    public int getLengthLimit() {return lengthLimit;}
    public int getTopK() {return topK;}
    public boolean isCalculationOnly() {return isCalculationOnly;}
    public Integer getRequestedMemorySize() {return requestedMemorySize;}
    public boolean isParallel() {return isParallel;}
    public String getServerAddress() {return serverAddress;}
    public Integer getPort() {return port;}

    // Setters
    public void setModelId(String modelId) {this.modelId = modelId;}
    public void setCalculationOnly(boolean calculationOnly) {isCalculationOnly = calculationOnly;}

    public String getConfigPath()
    {
        return modelId == null ? null : configRoot + "/" + modelId;
    }

    public String getModelPath()
    {
        return modelId == null ? null : downloadRoot + "/" + modelId;
    }

    private static String removeDoubleQuotes(String text)
    {
        if (text == null) return null;
        if (text.charAt(0) == '"') text = text.substring(1);
        if (text.charAt(text.length() - 1) == '"') text = text.substring(0, text.length() - 1);
        return text;
    }
}