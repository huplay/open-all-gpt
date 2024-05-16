package huplay.tokenizer;

import huplay.IdentifiedException;
import huplay.config.TokenizerConfig;
import huplay.tokenizer.gpt.GPT1Tokenizer;
import huplay.tokenizer.gpt.GPT2Tokenizer;
import huplay.tokenizer.sentencePiece.SentencePieceTokenizer;

public enum TokenizerType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    SENTENCE_PIECE,
    TIKTOKEN;

    public static Tokenizer getTokenizer(TokenizerConfig tokenizerConfig)
    {
        var type = tokenizerConfig.getTokenizerType();
        if (type == null)
        {
            throw new IdentifiedException("Tokenizer type isn't specified");
        }

        type = type.toUpperCase();
        var variant = "";
        if (type.contains("/"))
        {
            var index = type.indexOf("/");
            variant = type.substring(index + 1);
            type = type.substring(0, index);
        }

        var tokenizerType = TokenizerType.valueOf(type);
        return switch (tokenizerType)
        {
            case OPENAI_GPT_1       -> new GPT1Tokenizer(tokenizerConfig);
            case OPENAI_GPT_2       -> new GPT2Tokenizer(tokenizerConfig);
            case SENTENCE_PIECE     -> new SentencePieceTokenizer(tokenizerConfig, variant);
            default ->
                    throw new IdentifiedException("Unknown tokenizer type: " + type);
        };
    }
}
