package huplay.network.message.toServer.fromClient;

import huplay.network.message.BaseRequest;

public class QueryRequest extends BaseRequest<QueryResponse>
{
    private String modelId;
    private String sessionUUID;
    private int topK;
    private int maxLength;
    private String text;

    public QueryRequest() {} // Empty constructor for deserialization

    public QueryRequest(String modelId, String sessionUUID, String text, int topK, int maxLength)
    {
        this.modelId = modelId;
        this.sessionUUID = sessionUUID;
        this.topK = topK;
        this.maxLength = maxLength;
        this.text = text;
    }

    // Getters
    public String getModelId() {return modelId;}
    public String getSessionUUID() {return sessionUUID;}
    public int getTopK() {return topK;}
    public int getMaxLength() {return maxLength;}
    public String getText() {return text;}
}
