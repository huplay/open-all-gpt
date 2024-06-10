package network.message.toServer.fromClient;

import network.message.BasePollRequest;

public class PollQueryResultRequest extends BasePollRequest<PollQueryResultResponse>
{
    private String modelId;
    private String queryUUID;

    public PollQueryResultRequest() {} // Empty constructor for deserialization

    public PollQueryResultRequest(String modelId, String queryUUID)
    {
        this.modelId = modelId;
        this.queryUUID = queryUUID;
    }

    // Getters
    public String getModelId() {return modelId;}
    public String getQueryUUID() {return queryUUID;}
}
