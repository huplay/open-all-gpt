package huplay.network;

import huplay.network.message.BaseMessage;

public interface ProgressHandler
{
    boolean isReady(BaseMessage message);
}
