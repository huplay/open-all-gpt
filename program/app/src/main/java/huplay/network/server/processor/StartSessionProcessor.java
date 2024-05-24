package huplay.network.server.processor;

import huplay.network.message.toServer.fromClient.StartSessionResponse;

import java.util.UUID;

public class StartSessionProcessor
{
    public static StartSessionResponse process()
    {
        System.out.println("StartSessionRequest received.");
        return new StartSessionResponse(UUID.randomUUID().toString());
    }
}
