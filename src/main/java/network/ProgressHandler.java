package network;

import network.message.BaseMessage;

public interface ProgressHandler
{
    boolean isReady(BaseMessage message);
}
