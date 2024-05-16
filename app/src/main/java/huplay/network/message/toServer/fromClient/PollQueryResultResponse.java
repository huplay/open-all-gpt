package huplay.network.message.toServer.fromClient;

import huplay.network.message.BasePollResponse;
import huplay.tokenizer.Token;

import java.util.List;

public class PollQueryResultResponse extends BasePollResponse
{
    private String queryUUID;
    private List<Token> tokens;
    private String text;
    private boolean ready;

    public PollQueryResultResponse() {} // Empty constructor for deserialization

    public PollQueryResultResponse(String queryUUID, List<Token> tokens, String text, boolean isReady)
    {
        this.queryUUID = queryUUID;
        this.tokens = tokens;
        this.text = text;
        this.ready = isReady;
    }

    // Getters
    public String getQueryUUID() {return queryUUID;}
    public List<Token> getTokens() {return tokens;}
    public String getText() {return text;}
    public boolean getReady() {return ready;}

    @Override
    public boolean isStopPolling()
    {
        return ready;
    }

    @Override
    public String toString()
    {
        return "GetQueryResultResponse{" +
                "queryUUID='" + queryUUID + '\'' +
                ", tokens=" + tokens +
                ", text='" + text + '\'' +
                ", isReady=" + ready +
                '}';
    }
}
