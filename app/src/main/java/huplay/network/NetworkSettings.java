package huplay.network;

import java.io.File;
import java.io.PrintStream;

import static huplay.ui.ConsoleUtil.input;
import static huplay.ui.ConsoleUtil.intInput;
import static huplay.ui.TextUtil.equalsIgnoreCase;
import static huplay.ui.TextUtil.readInt;

public class NetworkSettings
{
    private static final String ARG_JOIN = "-join";
    private static final String ARG_PORT = "-port";

    // The root folder of the model configurations
    // The default is the "models", but it can be overridden by the OPEN_ALL_GPT_MODELS_ROOT environment variable
    private final String configRoot;

    // The root folder of the model parameters
    // The default is the "download", but it can be overridden by the OPEN_ALL_GPT_DOWNLOAD_ROOT environment variable
    private final String downloadRoot;

    // ModelId - the path of the selected model
    private String modelId;

    private final String serverHost;
    private final int serverPort;
    private final Address serverAddress;

    private final int selfPort;

    public NetworkSettings(String configRoot, String downloadRoot, String serverHost,
                           int serverPort, int selfPort)
    {
        this.configRoot = configRoot;
        this.downloadRoot = downloadRoot;
        this.serverHost = serverHost;
        this.serverPort = serverPort;
        this.selfPort = selfPort;

        this.serverAddress = new Address(Endpoint.SERVER, serverHost, serverPort);
    }

    public static NetworkSettings read(PrintStream OUT, String[] args, boolean hasSelfPort)
    {
        var file = new File("models");
        var configRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_MODELS_ROOT", file.getAbsolutePath());

        file = new File("download");
        var downloadRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_DOWNLOAD_ROOT", file.getAbsolutePath());

        configRoot = configRoot.replace("\\", "/");
        downloadRoot = downloadRoot.replace("\\", "/");

        String serverHost = null;
        var serverPort = -1;
        var selfPort = -1;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
            for (var arg : args)
            {
                var parts = arg.split("=");
                if (parts.length == 2)
                {
                    var key = parts[0];
                    var value = parts[1];

                    if (equalsIgnoreCase(key, ARG_JOIN))
                    {
                        var serverParts = value.split(":");

                        serverHost = serverParts[0];

                        if (serverParts.length > 1)
                        {
                            serverPort = readInt(serverParts[1], -1);
                        }
                    }
                    else if (equalsIgnoreCase(key, ARG_PORT))
                    {
                        selfPort = readInt(value, -1);
                    }
                }
            }
        }

        if (serverHost == null)
        {
            serverHost = input(OUT, "Server host: ");
        }

        if (serverPort < 0)
        {
            serverPort = intInput(OUT, "Server port: ", "Wrong number");
        }

        if (hasSelfPort && selfPort < 0)
        {
            selfPort = intInput(OUT, "Self port: ", "Wrong number");
        }
        else if (!hasSelfPort)
        {
            selfPort = -1;
        }

        return new NetworkSettings(configRoot, downloadRoot, serverHost, serverPort, selfPort);
    }

    // Getters
    public String getConfigRoot() {return configRoot;}
    public String getDownloadRoot() {return downloadRoot;}
    public String getModelId() {return modelId;}
    public String getServerHost() {return serverHost;}
    public Integer getServerPort() {return serverPort;}
    public Address getServerAddress() {return serverAddress;}
    public Integer getSelfPort() {return selfPort;}

    // Setter
    public void setModelId(String modelId) {this.modelId = modelId;}

    public String getConfigPath()
    {
        return modelId == null ? null : configRoot + "/" + modelId;
    }

    public String getModelPath()
    {
        return modelId == null ? null : downloadRoot + "/" + modelId;
    }
}
