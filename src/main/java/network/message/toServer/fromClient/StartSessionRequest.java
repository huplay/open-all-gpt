package network.message.toServer.fromClient;

import network.message.BaseRequest;

public class StartSessionRequest extends BaseRequest<StartSessionResponse>
{
    private String modelId;

    public StartSessionRequest() {} // Empty constructor for deserialization

    public StartSessionRequest(String modelId)
    {
        this.modelId = modelId;
    }

    // Getter
    public String getModelId() {return modelId;}
}
