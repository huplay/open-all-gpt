package huplay.network;

public enum Endpoint
{
    SERVER("/openAllGPT/server"),
    WORKER("/openAllGPT/worker");

    private final String context;

    Endpoint(String context)
    {
        this.context = context;
    }

    public String getContext()
    {
        return context;
    }
}
