package tokenizer;

import java.util.List;

public interface Tokenizer
{
    /**
     * Convert text to list of tokens
     */
    List<Integer> encode(String text);

    /**
     * Convert list of tokens to text
     */
    String decode(List<Integer> tokens);

    /**
     * Split the text to tokens (for displaying purposes)
     */
    List<Token> split(String text);

    Token getEndOfTextToken();
}
