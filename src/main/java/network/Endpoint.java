package network;

public enum Endpoint
{
    SERVER("/open-all-gpt"),
    WORKER("/open-all-gpt");

    private final String domain;

    Endpoint(String domain)
    {
        this.domain = domain;
    }

    public String getDomain()
    {
        return domain;
    }
}
