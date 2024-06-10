package network.message.toServer.fromClient;

import network.message.BasePollResponse;

/**
 * Message from client to server
 * Tells the server a user wants to use this model
 */
public class PollOpenModelResponse extends BasePollResponse
{
    private boolean ready;

    public PollOpenModelResponse() {} // Empty constructor for deserialization

    public PollOpenModelResponse(boolean isReady)
    {
        this.ready = isReady;
    }

    // Getter
    public boolean getReady() {return ready;}

    @Override
    public boolean isStopPolling() {return ready;}

    @Override
    public String toString()
    {
        return "OpenModelResponse{" +
                "ready=" + ready +
                '}';
    }
}
