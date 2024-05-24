package huplay.network;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Address
{
    private Endpoint endpoint;
    private String host;
    private int port;

    public Address() {} // Empty constructor for Json deserialization

    public Address(Endpoint endpoint, String host, int port)
    {
        this.endpoint = endpoint;
        this.host = host;
        this.port = port;
    }

    // Getters
    public Endpoint getEndpoint() {return endpoint;}
    public String getHost() {return host;}
    public int getPort() {return port;}

    @JsonIgnore
    public String getURL()
    {
        return "http://" + host + ":" + port + endpoint.getDomain();
    }

    @Override
    public String toString()
    {
        return endpoint + "/" + host + ':' + port;
    }
}
