package tokenizer.sentencePiece;

import app.IdentifiedException;
import config.TokenizerConfig;
import tokenizer.Token;
import tokenizer.Tokenizer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

import static tokenizer.sentencePiece.SentencePieceModel.ModelProto;
import static ui.TextUtil.equalsIgnoreCase;

/**
 * Java implementation of Google SentencePiece tokenizer (without training)
 * Original implementation: <a href="https://github.com/google/sentencepiece" />
 *
 * @author Hunor Szegi
 */
public class SentencePieceTokenizer implements Tokenizer
{
    private static final int START_OF_TEXT = 1;
    private static final String HEX_TOKEN_PREFIX = "<0x";
    private static final String HEX_TOKEN_SUFFIX = ">";

    protected List<String> vocabulary;
    protected List<Float> vocabularyScores;
    private final Map<String, Integer> vocabularyIndex = new HashMap<>();

    public SentencePieceTokenizer(TokenizerConfig tokenizerConfig, String variant)
    {
        if (equalsIgnoreCase(variant, "TINY"))
        {
            initTiny(tokenizerConfig);
        }
        else
        {
            initStandard(tokenizerConfig);
        }

        // Create the vocabulary index, to quickly find the id of a token
        for (var i = 0; i < vocabulary.size(); i++)
        {
            vocabularyIndex.put(vocabulary.get(i), i);
        }
    }

    private void initStandard(TokenizerConfig tokenizerConfig)
    {
        var tokenizerFile = tokenizerConfig.findFile("tokenizer.model");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile())
        {
            throw new IdentifiedException("SentencePiece tokenizer file is missing. (" + tokenizerFile.getName() + ")");
        }
        /*
            The tokenizer.model file (which contains the vocabulary and the scores) is a Protocol Buffer file.
            Protocol Buffer (Protobuf) is a cross-platform data serialization technique created by Google:
            https://en.wikipedia.org/wiki/Protocol_Buffers

            It needs a .proto file to describe the structure of the data. For SentencePiece you can found it here:
            https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
            (It is stored in this repo at the "resources" folder as well, but it needed only once.)

            Based on the .proto file you can generate source code (in different languages)
            to store and read/write data using the protobuf app (protoc.exe). Download protobuf:
            https://github.com/protocolbuffers/protobuf/releases/tag/v26.1

            The SentencePieceModel.java was generated using this prompt:
            protoc.exe -I=. --java_out=. sentencepiece_model.proto
            (It is slightly modified to put into the correct package and renamed to camelcase.)
         */
        try
        {
            var model = ModelProto.newBuilder().mergeFrom(new FileInputStream(tokenizerFile)).build();

            var sentencesPieces = model.getPiecesList();

            this.vocabulary = new ArrayList<>(sentencesPieces.size());
            this.vocabularyScores = new ArrayList<>(sentencesPieces.size());

            for (var sentencePiece : sentencesPieces)
            {
                vocabulary.add(sentencePiece.getPiece());
                vocabularyScores.add(sentencePiece.getScore());
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("SentencePiece tokenizer.model reading error.", e);
        }
    }

    private void initTiny(TokenizerConfig tokenizerConfig)
    {
        this.vocabulary = new ArrayList<>(32000);
        this.vocabularyScores = new ArrayList<>(32000);

        var tokenizerFile = tokenizerConfig.findFile("tokenizer.bin");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile())
        {
            throw new IdentifiedException("SentencePiece tokenizer file is missing. (" + tokenizerFile.getName() + ")");
        }

        // Read the vocabulary from the binary config file
        var configFilePath = Paths.get(tokenizerFile.getAbsolutePath());
        try (var channel = FileChannel.open(configFilePath, StandardOpenOption.READ))
        {
            var buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // The first integer contains the maximum token length (we don't need this info)
            buffer.getInt();

            // Iterate over on all tokens
            while(buffer.hasRemaining())
            {
                // Read vocabulary score
                vocabularyScores.add(buffer.getFloat());

                // Read token length
                var tokenLength = buffer.getInt();

                // Read token
                byte[] bytes = new byte[tokenLength];
                buffer.get(bytes);

                vocabulary.add(new String(bytes, StandardCharsets.UTF_8));
            }
        }
        catch (Exception e)
        {
            throw new IdentifiedException("SentencePiece tokenizer.bin reading error: " + e.getMessage());
        }
    }

    private int getId(String text)
    {
        return vocabularyIndex.getOrDefault(text, -1);
    }

    @Override
    public List<Integer> encode(String text)
    {
        if (text == null)
        {
            return Collections.singletonList(0);
        }
        else if (text.isEmpty())
        {
            return Collections.singletonList(START_OF_TEXT);
        }

        List<Integer> tokens = new ArrayList<>();

        tokens.add(START_OF_TEXT);

        // First encode the text as a sequence of individual unicode characters
        int codePoint;
        for (var i = 0; i < text.length(); i += Character.charCount(codePoint))
        {
            codePoint = text.codePointAt(i);
            var character = Character.toString(codePoint);

            var id = getId(character);

            if (id != -1)
            {
                // The character is in the vocabulary
                tokens.add(id);
            }
            else
            {
                // Rare unicode character, not in the vocabulary. Encode the character as a sequence of bytes
                // (All single-byte (0-255) and a lot of multibyte characters are in the vocabulary, but not all)
                // (The single-byte characters are stored after the <unk>, <s>, </s>, that's why the "+ 3")
                for (var b : character.getBytes(StandardCharsets.UTF_8))
                {
                    tokens.add(Byte.toUnsignedInt(b) + 3);
                }
            }
        }

        // Merge pairs of encoded characters into a single token, using the scores to find the best merges
        while (true)
        {
            var bestScore = -1e10f;
            var bestId = -1;
            var bestIndex = -1;

            // Iterate over on the tokens (without the last one)
            // Find the best merge of token[i] and token[i + 1]
            for (var i = 0; i < tokens.size() - 1; i++)
            {
                var id1 = tokens.get(i);
                var id2 = tokens.get(i);

                if (id1 >= 0 && id2 >= 0 && id1 < vocabulary.size() && id2 < vocabulary.size())
                {
                    var mergedText = vocabulary.get(id1) + vocabulary.get(id2);
                    var id = getId(mergedText);

                    if (id != -1 && vocabularyScores.get(id) > bestScore)
                    {
                        // If this text exists in the vocabulary, and has better score as the best, record it
                        bestScore = vocabularyScores.get(id);
                        bestId = id;
                        bestIndex = i;
                    }
                }
            }

            if (bestIndex == -1)
            {
                // There's nothing to merge, we are ready
                break;
            }

            // Do the best merge
            tokens.set(bestIndex, bestId);
            tokens.remove(bestIndex + 1);
        }

        return tokens;
    }

    @Override
    public String decode(List<Integer> tokens)
    {
        var buffer = new StringBuilder();

        var prevToken = 0;
        for (var token : tokens)
        {
            buffer.append(decodeToken(prevToken, token));
            prevToken = token;
        }

        return buffer.toString();
    }

    private String decodeToken(int prevToken, Integer token)
    {
        var text = vocabulary.get(token);

        if (text == null)
        {
            return "<ERROR>";
        }

        if (text.length() == 6 && text.startsWith(HEX_TOKEN_PREFIX) && text.endsWith(HEX_TOKEN_SUFFIX))
        {
            // If it is a hex token (in the format of <0xNN>), convert the hex value (NN) to character
            var hex = text.substring(HEX_TOKEN_PREFIX.length(), HEX_TOKEN_PREFIX.length() + 2);
            text = Character.toString(Integer.parseInt(hex, 16));
        }
        else if (prevToken == START_OF_TEXT && text.charAt(0) == ' ')
        {
            // If the token is the first one with leading space, remove the space
            text = text.substring(1);
        }

        return text;
    }

    @Override
    public List<Token> split(String text)
    {
        if (text == null || text.isEmpty()) return Collections.emptyList();

        List<Token> tokens = new ArrayList<>();

        // First encode the text as a sequence of individual unicode characters
        int codePoint;
        for (var i = 0; i < text.length(); i += Character.charCount(codePoint))
        {
            codePoint = text.codePointAt(i);
            var character = Character.toString(codePoint);

            var id = getId(character);

            if (id != -1)
            {
                // The character is in the vocabulary
                tokens.add(new Token(id, character));
            }
            else
            {
                // Rare unicode character, not in the vocabulary. Encode the character as a sequence of bytes
                // (All single-byte (0-255) and a lot of multibyte characters are in the vocabulary, but not all)
                // (The single-byte characters are stored after the <unk>, <s>, </s>, that's why the "+ 3")
                for (var b : character.getBytes(StandardCharsets.UTF_8))
                {
                    tokens.add(new Token(id, "" + (Byte.toUnsignedInt(b) + 3)));
                }
            }
        }

        // Merge pairs of encoded characters into a single token, using the scores to find the best merges
        while (true)
        {
            var bestScore = -1e10f;
            var bestId = -1;
            var bestText = "";
            var bestIndex = -1;

            // Iterate over on the tokens (without the last one)
            // Find the best merge of token[i] and token[i + 1]
            for (var i = 0; i < tokens.size() - 1; i++)
            {
                var id1 = tokens.get(i).getId();
                var id2 = tokens.get(i + 1).getId();

                if (id1 >= 0 && id2 >= 0 && id1 < vocabulary.size() - 1 && id2 < vocabulary.size() - 1)
                {
                    var mergedText = vocabulary.get(id1) + vocabulary.get(id2);
                    var id = getId(mergedText);

                    if (id != -1 && vocabularyScores.get(id) > bestScore)
                    {
                        // If this text exists in the vocabulary, and has better score as the best, record it
                        bestScore = vocabularyScores.get(id);
                        bestId = id;
                        bestText = mergedText;
                        bestIndex = i;
                    }
                }
            }

            if (bestIndex == -1)
            {
                // There's nothing to merge, we are ready
                break;
            }

            // Do the best merge
            tokens.set(bestIndex, new Token(bestId, bestText));
            tokens.remove(bestIndex + 1);
        }

        return tokens;
    }

    @Override
    public Token getEndOfTextToken()
    {
        return new Token(2, "");
    }
}