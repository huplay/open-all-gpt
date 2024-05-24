package huplay;

import huplay.tokenizer.Tokenizer;

import java.util.List;

public interface Flow
{
    List<Integer> process(List<Integer> inputTokens, int pos);

    void clear();

    Tokenizer getTokenizer();

    int getEndOfTextToken();
}
