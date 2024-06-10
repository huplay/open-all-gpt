package network.info.input;

public class TokenInput implements Input
{
    private int pos;
    private int token;

    public TokenInput()
    {
    }

    public TokenInput(int pos, int token)
    {
        this.pos = pos;
        this.token = token;
    }

    // Getters
    public int getPos() {return pos;}
    public int getToken() {return token;}

    @Override
    public String toString()
    {
        return "TokenInput{" +
                "pos=" + pos +
                ", token=" + token +
                '}';
    }
}
