package network.message;

import com.fasterxml.jackson.annotation.JsonIgnore;

public abstract class BasePollResponse extends BaseResponse
{
    @JsonIgnore
    abstract protected boolean isStopPolling();
}
