package huplay.tokenizer.gpt;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.config.TokenizerConfig;
import huplay.tokenizer.Token;
import huplay.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Tokenizer which fully compatible with the OpenAI GPT-2 and GPT-3 tokenizer (used for other models as well)
 *
 * @author Hunor Szegi
 */
public class GPT2Tokenizer implements Tokenizer
{
    private final Map<Character, Byte> charEncoding = new HashMap<>(256);
    private final Map<Integer, Character> charDecoding = new HashMap<>(256);

    private final Map<String, Integer> tokenEncoding;
    private final Map<Integer, String> tokenDecoding = new HashMap<>(50257);

    private final Map<Pair, Integer> merges;

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public GPT2Tokenizer(TokenizerConfig tokenizerConfig)
    {
        addCharRange(0, 256, 288);
        addCharRange(33, 33, 126);
        addCharRange(127, 289, 322);
        addCharRange(161, 161, 172);
        addCharRange(173, 323, 323);
        addCharRange(174, 174, 255);

        var vocabularyFile = tokenizerConfig.findFile("vocab.json");
        if (!vocabularyFile.exists() || !vocabularyFile.isFile())
        {
            throw new IdentifiedException("GPT-2 tokenizer vocabulary file is missing. (" + vocabularyFile.getName() + ")");
        }

        try
        {
            var typeRef = new TypeReference<Map<String, Integer>>() {};
            tokenEncoding = new ObjectMapper().readValue(vocabularyFile, typeRef);

            tokenEncoding.forEach((key, value) -> tokenDecoding.put(value, key));
        }
        catch (IOException e)
        {
            throw new IdentifiedException("GPT-2 tokenizer vocabulary reading error.", e);
        }

        var mergesFile = tokenizerConfig.findFile("merges.txt");
        if (!mergesFile.exists() || !mergesFile.isFile())
        {
            throw new IdentifiedException("GPT-2 tokenizer merges file is missing. (" + mergesFile.getName() + ")");
        }

        merges = MergesReader.readMergesFile(mergesFile, true);
    }

    @Override
    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        var unicodeText = splitToUnicode(text);

        var result = new ArrayList<Integer>();

        for (var word : unicodeText)
        {
            for (var token : BytePairEncoding.encode(word, merges).split(" "))
            {
                var value = tokenEncoding.get(token);
                if (value != null)
                {
                    result.add(value);
                }
            }
        }

        return result;
    }

    private List<String> splitToUnicode(String text)
    {
        var unicodeText = new ArrayList<String>();

        var matcher = pattern.matcher(text);
        while (matcher.find())
        {
            var match = new StringBuilder();

            var buffer = StandardCharsets.UTF_8.encode(matcher.group());
            while (buffer.hasRemaining())
            {
                int value = buffer.get();
                if (value < 0) value = value & 0xff;
                match.append(charDecoding.get(value));
            }

            unicodeText.add(match.toString());
        }

        return unicodeText;
    }

    @Override
    public String decode(List<Integer> tokens)
    {
        var textBuilder = new StringBuilder();
        for (var token : tokens)
        {
            textBuilder.append(tokenDecoding.get(token));
        }
        var text = textBuilder.toString();

        var bytes = new byte[text.length()];
        for (var i = 0; i < text.length(); i++)
        {
            bytes[i] = charEncoding.get(text.charAt(i));
        }

        return new String(bytes, StandardCharsets.UTF_8);
    }

    @Override
    public List<Token> split(String text)
    {
        if (text == null || text.isEmpty()) return Collections.emptyList();

        var unicodeText = splitToUnicode(text);

        var result = new ArrayList<Token>();

        for (var word : unicodeText)
        {
            for (var token : BytePairEncoding.encode(word, merges).split(" "))
            {
                var id = tokenEncoding.get(token);
                if (id != null)
                {
                    result.add(new Token(id, token.replace("Ġ", " ")));
                }
            }
        }

        return result;
    }

    @Override
    public Token getEndOfTextToken()
    {
        return new Token(50256, "");
    }

    private void addCharRange(int pos, int firstChar, int lastChar)
    {
        for (var i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put((char) i, (byte)pos);
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }
}
