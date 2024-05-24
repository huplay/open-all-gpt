package huplay.network;

import huplay.Flow;
import huplay.config.Arguments;
import huplay.config.ModelConfig;
import huplay.tokenizer.Tokenizer;

import java.io.IOException;
import java.net.InetAddress;
import java.util.List;

public class NetworkFlow implements Flow
{
    private final ModelConfig modelConfig;
    private final Tokenizer tokenizer;
    private final String serverAddress;
    private final int serverPort;
    private final String selfAddress;
    private final int selfPort;

    public NetworkFlow(Arguments arguments, ModelConfig modelConfig, Tokenizer tokenizer) throws IOException
    {
        this.modelConfig = modelConfig;
        this.tokenizer = tokenizer;

        // TODO: Check, move to arguments
        var targetAddress = arguments.getServerAddress();
        var parts = targetAddress.split(":");
        this.serverAddress = parts[0];
        this.serverPort = Integer.parseInt(parts[1]);
        System.out.println("Server: " + serverAddress + ", port: " + serverPort);

        var selfAddress = InetAddress.getLocalHost().getHostAddress();
        var selfParts = selfAddress.split(":");
        this.selfAddress = selfParts[0];
        this.selfPort = arguments.getPort();
        System.out.println("Self: " + selfAddress + ", port: " + selfPort);

        /*OpenModelMessage openModelMessage = new OpenModelMessage(selfAddress, selfPort, arguments.getRelativePath());
        System.out.println("request: " + openModelMessage);

        openModelMessage.send(serverAddress, serverPort);*/

        // TODO: Start thread which can communicate with the server

    }

    @Override
    public void clear()
    {

    }

    @Override
    public List<Integer> process(List<Integer> inputTokens, int pos)
    {
        return List.of();
    }

    @Override
    public Tokenizer getTokenizer()
    {
        return null;
    }

    @Override
    public int getEndOfTextToken()
    {
        return 0;
    }
}
