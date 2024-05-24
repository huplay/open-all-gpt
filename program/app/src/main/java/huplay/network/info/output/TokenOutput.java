package huplay.network.info.output;

public class TokenOutput implements Output
{
    private int token;

    public TokenOutput() {} // Empty constructor for deserialization

    public TokenOutput(int token)
    {
        this.token = token;
    }

    // Getter
    public int getToken() {return token;}

    @Override
    public String toString()
    {
        return "TokenOutput{" +
                "token=" + token +
                '}';
    }
}
