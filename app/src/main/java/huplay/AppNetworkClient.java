package huplay;

import huplay.config.RepoConfig;
import huplay.network.message.toServer.fromClient.PollQueryResult;
import huplay.network.message.toServer.fromClient.PollOpenModel;
import huplay.network.message.toServer.fromClient.QueryRequest;
import huplay.network.message.toServer.fromClient.StartSessionRequest;
import huplay.network.NetworkSettings;
import huplay.util.Util;

import java.io.*;
import java.util.*;

import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.ConsoleUtil.input;
import static huplay.ui.Logo.logo;
import static huplay.ui.ModelSelector.selectModel;
import static huplay.ui.TextUtil.toCenter;

public class AppNetworkClient
{
    public static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static void main(String... args)
    {
        try
        {
            logo(OUT,"Open All GPT", "CWgY-CWY-bgW", 'W');
            OUT.println(toCenter("Network Client\n", 60));

            new AppNetworkClient().start(args);
        }
        catch (Exception e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
    }

    private void start(String... args) throws Exception
    {
        // Read arguments
        var settings = NetworkSettings.read(OUT, args, false);
        var server = settings.getServerAddress();

        // TODO: Potentially server can send the configuration
        // (Then it is enough to refresh the server, and there won't be any differences between configs)
        // ClientJoined message

        // Select model
        var modelId = selectModel(OUT, settings.getConfigRoot());
        settings.setRelativePath(modelId);

        OUT.print("Connecting to server to open model... ");
        // TODO: Display progress using ProgressHandler
        var openModel = new PollOpenModel(modelId);
        openModel.poll(server);
        OUT.println("DONE");

        var startSessionRequest = new StartSessionRequest();
        var startSessionResponse = startSessionRequest.send(server);
        var sessionUUID = startSessionResponse.getSessionUUID();

        while (true)
        {
            // Read the input text
            var inputText = input(OUT, "\n\nInput text: ");

            var queryRequest = new QueryRequest(modelId, sessionUUID, inputText, 40, 25);
            var queryResponse = queryRequest.send(server);

            var queryUUID = queryResponse.getQueryUUID();
            var split = queryResponse.getTokens();

            OUT.print("            ");
            for (var i = 0; i < split.size(); i++)
            {
                var color = i % 2 == 0 ? "\033[32m" : "\033[33m";
                OUT.print(color + split.get(i).getText().replace("\n", "").replace("\r", ""));
            }
            OUT.println("\033[0m");

            var queryResult = new PollQueryResult(modelId, queryUUID);
            var result = queryResult.poll(server); // TODO: Add ProgressHelper

            OUT.println(result.getText());
        }
    }

    public static List<String> checkFiles(RepoConfig tokenizerConfig, String modelPath)
    {
        var missingFiles = new ArrayList<String>();

        for (var fileName : tokenizerConfig.getFiles())
        {
            var path = modelPath + "/" + fileName;

            var file = new File(path);
            if (!file.exists())
            {
                missingFiles.add(fileName);
            }
        }

        return missingFiles;
    }
}
