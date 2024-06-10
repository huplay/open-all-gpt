package network.message.toServer.fromClient;

import network.info.Models;
import network.message.BaseResponse;

import java.util.Map;

public class ClientJoinedResponse extends BaseResponse
{
    private Map<String, Models> models;

    public ClientJoinedResponse()
    {
    }

    public ClientJoinedResponse(Map<String, Models> models)
    {
        this.models = models;
    }

    // Getter
    public Map<String, Models> getModels() {return models;}

    @Override
    public String toString()
    {
        return "ClientJoinedResponse{" +
                "models=" + models +
                '}';
    }
}
