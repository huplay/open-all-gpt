package huplay.network.message.toServer.fromClient;

import huplay.network.message.BasePollRequest;

public class PollQueryResult extends BasePollRequest<PollQueryResultResponse>
{
    private String modelId;
    private String queryUUID;

    public PollQueryResult() {} // Empty constructor for deserialization

    public PollQueryResult(String modelId, String queryUUID)
    {
        this.modelId = modelId;
        this.queryUUID = queryUUID;
    }

    // Getters
    public String getModelId() {return modelId;}
    public String getQueryUUID() {return queryUUID;}
}
