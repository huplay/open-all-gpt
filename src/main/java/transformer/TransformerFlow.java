package transformer;

import tokenizer.Tokenizer;

import java.util.List;

public interface TransformerFlow
{
    String getFlowType();

    List<Integer> process(List<Integer> inputTokens, int posOffset);

    void clear();

    Tokenizer getTokenizer();

    int getEndOfTextToken();

    long getParameterSize();
}
