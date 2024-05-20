package huplay;

import huplay.network.message.toServer.fromClient.*;
import huplay.network.NetworkSettings;
import huplay.ui.ModelSelector;
import huplay.util.Util;

import java.io.*;

import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.ConsoleUtil.input;
import static huplay.ui.Logo.logo;
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

        // Send ClientJoined message and get back the tree of models
        var clientJoinedResponse = new ClientJoinedRequest().send(server);
        var models = clientJoinedResponse.getModels();

        // Select model
        var modelSelector = new ModelSelector(OUT, models);
        var modelId = modelSelector.select();
        settings.setModelId(modelId);

        OUT.print("Connecting to server to open model... ");
        // TODO: Display progress using ProgressHandler
        var openModel = new PollOpenModelRequest(modelId);
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

            var queryResult = new PollQueryResultRequest(modelId, queryUUID);
            var result = queryResult.poll(server); // TODO: Add ProgressHelper

            OUT.println(result.getText());
        }
    }
}
