package huplay;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.config.ParameterType;
import huplay.parameters.safetensors.SafetensorsReader;
import huplay.parameters.safetensors.SafetensorsModel;
import huplay.parameters.StandardParameterLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static huplay.parameters.safetensors.SafetensorsModel.TensorModel;
import static huplay.math.TypeConversionUtility.toLittleEndian;

/**
 * Generates test .safetensors file to use in unit tests
 * It uses a subset of the values of a .safetensors file (not random, to make the scale real)
 * (Supports only 1 or 2 dimensions, float32 values, and uses a fixed metadata)
 *
 * @author Hunor Szegi
 */
public class GenerateTestSafetensors
{
    public static void main(String... args) throws IOException
    {
        var testEntries = getTestEntriesGPT1();
        //var testEntries = getTestEntriesGPT2();
        //var testEntries = getTestEntriesGPTNEO();

        generate("d:/test", "model.safetensors", "test.safetensors", testEntries);
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPT1()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var feedForwardSize = hiddenSize * 4;

        var testEntries = new LinkedHashMap<String, int[]>();

        testEntries.put("tokens_embed.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("positions_embed.weight", new int[] {contextSize, hiddenSize});

        testEntries.put("h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("h.0.attn.c_attn.weight", new int[] {hiddenSize, hiddenSize * 3});
        testEntries.put("h.0.attn.c_attn.bias", new int[] {hiddenSize * 3});
        testEntries.put("h.0.attn.c_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("h.0.attn.c_proj.bias", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPT2()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var feedForwardSize = hiddenSize * 4;

        var testEntries = new LinkedHashMap<String, int[]>();

        testEntries.put("wte.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("wpe.weight", new int[] {contextSize, hiddenSize});
        testEntries.put("ln_f.weight", new int[] {hiddenSize});
        testEntries.put("ln_f.bias", new int[] {hiddenSize});

        testEntries.put("h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("h.0.attn.c_attn.weight", new int[] {hiddenSize, hiddenSize * 3});
        testEntries.put("h.0.attn.c_attn.bias", new int[] {hiddenSize * 3});
        testEntries.put("h.0.attn.c_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("h.0.attn.c_proj.bias", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPTNEO()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var feedForwardSize = hiddenSize * 4;

        var testEntries = new LinkedHashMap<String, int[]>();

        testEntries.put("transformer.wte.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("transformer.wpe.weight", new int[] {contextSize, hiddenSize});
        testEntries.put("transformer.ln_f.weight", new int[] {hiddenSize});
        testEntries.put("transformer.ln_f.bias", new int[] {hiddenSize});

        testEntries.put("transformer.h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.q_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.k_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.v_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.out_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.out_proj.bias", new int[] {hiddenSize});

        testEntries.put("transformer.h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("transformer.h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("transformer.h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("transformer.h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static void generate(String path, String inputSafetensors, String outputSafetensors,
                                 LinkedHashMap<String, int[]> testEntries) throws IOException
    {
        // Create JSON header from the specified test entries
        var header = getHeader(testEntries);
        var headerSize = header.length();

        // Read the source safetensors file
        var reader = new SafetensorsReader(path);
        reader.readSafetensorsHeader(path + "/" + inputSafetensors);

        var parameterLoader = new StandardParameterLoader(null);

        // Write the test file
        var output = new File(path + "/" + outputSafetensors);
        try (var out = new DataOutputStream(new FileOutputStream(output)))
        {
            // Write the length of the header (long)
            out.writeLong(toLittleEndian(headerSize));

            // Write the header (string)
            out.write(header.getBytes(StandardCharsets.UTF_8));

            // Write the tensor values (float)
            for (var entry : testEntries.entrySet())
            {
                var shape = entry.getValue();
                if (shape.length == 1)
                {
                    // Write a vector
                    var values = parameterLoader.readVector(reader, entry.getKey(), shape[0]);
                    for (var i = 0; i < values.size(); i++)
                    {
                        var value = values.get(i);
                        out.writeFloat(toLittleEndian(value));
                    }
                }
                else if (shape.length == 2)
                {
                    // Write a matrix
                    var values = parameterLoader.readMatrix(reader, ParameterType.TEST, entry.getKey(), shape[0], shape[1]);
                    for (var row : values.getVectorArray())
                    {
                        for (var i = 0; i < row.size(); i++)
                        {
                            var value = row.get(i);
                            out.writeFloat(toLittleEndian(value));
                        }
                    }
                }
            }
        }
    }

    private static String getHeader(Map<String, int[]> descriptions)
    {
        var startOffset = 0L;

        var metadata = new HashMap<String, String>();
        metadata.put("format", "pt");

        var safetensorsModel = new SafetensorsModel(metadata);

        for (var description : descriptions.entrySet())
        {
            var dims = description.getValue();

            var size = dims[0];
            for (var i = 1; i < dims.length; i++)
            {
                size *= dims[i];
            }

            var shape = Arrays.stream(dims).boxed().collect(Collectors.toList());

            var endOffset = startOffset + size * 4L;

            var tensorModel = new TensorModel("F32", shape, startOffset, endOffset);
            safetensorsModel.addTensor(description.getKey(), tensorModel);

            startOffset = endOffset;
        }

        try
        {
            return new ObjectMapper().writeValueAsString(safetensorsModel);
        }
        catch (JsonProcessingException e)
        {
            return "";
        }
    }
}
