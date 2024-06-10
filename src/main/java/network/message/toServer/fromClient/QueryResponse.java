package network.message.toServer.fromClient;

import network.message.BaseResponse;
import tokenizer.Token;

import java.util.List;

public class QueryResponse extends BaseResponse
{
    private String queryUUID;
    private List<Token> tokens;

    public QueryResponse() {} // Empty constructor for deserialization

    public QueryResponse(String queryUUID, List<Token> tokens)
    {
        this.queryUUID = queryUUID;
        this.tokens = tokens;
    }

    // Getters
    public String getQueryUUID() {return queryUUID;}
    public List<Token> getTokens() {return tokens;}
}
