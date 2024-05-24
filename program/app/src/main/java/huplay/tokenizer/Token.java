package huplay.tokenizer;

public class Token
{
    private int id;
    private String text;

    public Token() {} // Empty constructor for deserialization

    public Token(int id, String text)
    {
        this.id = id;
        this.text = text;
    }

    // Getters, setters
    public int getId() {return id;}
    public String getText() {return text;}
    public void setText(String text) {this.text = text;}
}
