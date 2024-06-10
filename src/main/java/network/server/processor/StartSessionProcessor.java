package network.server.processor;

import network.message.toServer.fromClient.StartSessionResponse;

import java.util.UUID;

public class StartSessionProcessor
{
    public static StartSessionResponse process()
    {
        System.out.println("StartSessionRequest received.");
        return new StartSessionResponse(UUID.randomUUID().toString());
    }
}
