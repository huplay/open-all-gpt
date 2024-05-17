package huplay.network;

public enum Endpoint
{
    SERVER("/open-all-gpt/server"),
    WORKER("/open-all-gpt/worker");

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
