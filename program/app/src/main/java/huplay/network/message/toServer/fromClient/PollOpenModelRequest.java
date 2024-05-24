package huplay.network.message.toServer.fromClient;

import huplay.network.message.BasePollRequest;

/**
 * Message from client to server
 * Tells the server a user wants to use this model
 */
public class PollOpenModelRequest extends BasePollRequest<PollOpenModelResponse>
{
    private String modelId;

    public PollOpenModelRequest() {} // Empty constructor for deserialization

    public PollOpenModelRequest(String modelId)
    {
        this.modelId = modelId;
    }

    // Getter
    public String getModelId() {return modelId;}
}
