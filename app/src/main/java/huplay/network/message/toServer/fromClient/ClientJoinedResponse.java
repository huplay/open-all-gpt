package huplay.network.message.toServer.fromClient;

import huplay.network.info.Models;
import huplay.network.message.BaseResponse;

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
