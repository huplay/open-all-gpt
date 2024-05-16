package huplay.network.message.toServer.fromClient;

import huplay.network.message.BasePollRequest;

/**
 * Message from client to server
 * Tells the server a user wants to use this model
 */
public class PollOpenModel extends BasePollRequest<PollOpenModelResponse>
{
    private String modelId;

    public PollOpenModel() {} // Empty constructor for deserialization

    public PollOpenModel(String modelId)
    {
        this.modelId = modelId;
    }

    // Getter
    public String getModelId() {return modelId;}
}
