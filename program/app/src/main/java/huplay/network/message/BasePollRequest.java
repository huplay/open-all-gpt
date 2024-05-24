package huplay.network.message;

import huplay.network.Address;

import java.io.IOException;

public abstract class BasePollRequest<T extends BasePollResponse> extends BaseRequest<T>
{
    private int attempt;

    public T poll(Address target) throws InterruptedException, IOException
    {
        while (true)
        {
            T response = send(target);

            if (response.isStopPolling())
            {
                return response;
            }

            Thread.sleep(1000);
            attempt++;
        }
    }

    // Getter, setter
    public int getAttempt() {return attempt;}
    public void setAttempt(int attempt) {this.attempt = attempt;}
}
