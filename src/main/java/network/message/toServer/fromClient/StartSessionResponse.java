package network.message.toServer.fromClient;

import network.message.BaseResponse;

public class StartSessionResponse extends BaseResponse
{
    private String sessionUUID;

    public StartSessionResponse() {} // Empty constructor for deserialization

    public StartSessionResponse(String sessionUUID)
    {
        this.sessionUUID = sessionUUID;
    }

    // Getter
    public String getSessionUUID() {return sessionUUID;}
}
