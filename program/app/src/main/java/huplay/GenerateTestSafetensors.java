package huplay;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.config.ParameterType;
import huplay.parameters.safetensors.SafetensorsDataType;
import huplay.parameters.safetensors.SafetensorsReader;
import huplay.parameters.safetensors.SafetensorsModel;
import huplay.parameters.StandardParameterLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

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
    private static final String VECTOR_PARAMETER = "vector";
    private static final String MATRIX_PARAMETER = "matrix";
    private static final String BOOL_ARRAY_2D_PARAMETER = "boolArray2D";
    
    private record TestParameter(String id, String type, int rows, int cols, List<Integer> shape) {}

    // TODO: Override output file (now the previous should be deleted)
    // TODO: Data type could be configured

    public static void main(String... args) throws IOException
    {
        //var testEntries = getTestEntriesGPT1();
        //var testEntries = getTestEntriesGPT2();
        //var testEntries = getTestEntriesGPTNEO();
        var testEntries = getTestEntriesGPTJ();

        generate("d:/test", "model.safetensors", "test.safetensors", testEntries);
    }
    
    private static List<TestParameter> getTestEntriesGPT1()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var intermediateSize = hiddenSize * 4;

        var testEntries = new ArrayList<TestParameter>();

        addMatrixParameter("tokens_embed.weight", tokenCount, hiddenSize, testEntries);
        addMatrixParameter("positions_embed.weight", contextSize, hiddenSize, testEntries);

        addVectorParameter("h.0.ln_1.weight", hiddenSize, testEntries);
        addVectorParameter("h.0.ln_1.bias", hiddenSize, testEntries);
        addMatrixParameter("h.0.attn.c_attn.weight", hiddenSize, hiddenSize * 3, testEntries);
        addVectorParameter("h.0.attn.c_attn.bias", hiddenSize * 3, testEntries);
        addMatrixParameter("h.0.attn.c_proj.weight", hiddenSize, hiddenSize, testEntries);
        addVectorParameter("h.0.attn.c_proj.bias", hiddenSize, testEntries);

        addVectorParameter("h.0.ln_2.weight", hiddenSize, testEntries);
        addVectorParameter("h.0.ln_2.bias", hiddenSize, testEntries);
        addMatrixParameter("h.0.mlp.c_fc.weight", hiddenSize, intermediateSize, testEntries);
        addVectorParameter("h.0.mlp.c_fc.bias", intermediateSize, testEntries);
        addMatrixParameter("h.0.mlp.c_proj.weight", intermediateSize, hiddenSize, testEntries);
        addVectorParameter("h.0.mlp.c_proj.bias", hiddenSize, testEntries);

        return testEntries;
    }

    private static List<TestParameter> getTestEntriesGPT2()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var intermediateSize = hiddenSize * 4;

        var testEntries = new ArrayList<TestParameter>();

        addMatrixParameter("wte.weight", tokenCount, hiddenSize, testEntries);
        addMatrixParameter("wpe.weight", contextSize, hiddenSize, testEntries);
        addVectorParameter("ln_f.weight", hiddenSize, testEntries);
        addVectorParameter("ln_f.bias", hiddenSize, testEntries);

        addVectorParameter("h.0.ln_1.weight", hiddenSize, testEntries);
        addVectorParameter("h.0.ln_1.bias", hiddenSize, testEntries);
        addMatrixParameter("h.0.attn.c_attn.weight", hiddenSize, hiddenSize * 3, testEntries);
        addVectorParameter("h.0.attn.c_attn.bias", hiddenSize * 3, testEntries);
        addMatrixParameter("h.0.attn.c_proj.weight", hiddenSize, hiddenSize, testEntries);
        addVectorParameter("h.0.attn.c_proj.bias", hiddenSize, testEntries);

        addVectorParameter("h.0.ln_2.weight", hiddenSize, testEntries);
        addVectorParameter("h.0.ln_2.bias", hiddenSize, testEntries);
        addMatrixParameter("h.0.mlp.c_fc.weight", hiddenSize, intermediateSize, testEntries);
        addVectorParameter("h.0.mlp.c_fc.bias", intermediateSize, testEntries);
        addMatrixParameter("h.0.mlp.c_proj.weight", intermediateSize, hiddenSize, testEntries);
        addVectorParameter("h.0.mlp.c_proj.bias", hiddenSize, testEntries);

        return testEntries;
    }

    private static List<TestParameter> getTestEntriesGPTNEO()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var contextSize = 10;
        var intermediateSize = hiddenSize * 4;

        var testEntries = new ArrayList<TestParameter>();

        addMatrixParameter("transformer.wte.weight", tokenCount, hiddenSize, testEntries);
        addMatrixParameter("transformer.wpe.weight", contextSize, hiddenSize, testEntries);
        addVectorParameter("transformer.ln_f.weight", hiddenSize, testEntries);
        addVectorParameter("transformer.ln_f.bias", hiddenSize, testEntries);

        addVectorParameter("transformer.h.0.ln_1.weight", hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.ln_1.bias", hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.attention.q_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.attention.k_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.attention.v_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.attention.out_proj.weight", hiddenSize, hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.attn.attention.out_proj.bias", hiddenSize, testEntries);

        addVectorParameter("transformer.h.0.ln_2.weight", hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.ln_2.bias", hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.mlp.c_fc.weight", hiddenSize, intermediateSize, testEntries);
        addVectorParameter("transformer.h.0.mlp.c_fc.bias", intermediateSize, testEntries);
        addMatrixParameter("transformer.h.0.mlp.c_proj.weight", intermediateSize, hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.mlp.c_proj.bias", hiddenSize, testEntries);

        return testEntries;
    }

    private static List<TestParameter> getTestEntriesGPTJ()
    {
        var tokenCount = 10;
        var hiddenSize = 12;
        var intermediateSize = hiddenSize * 4;

        var testEntries = new ArrayList<TestParameter>();

        addMatrixParameter("transformer.wte.weight", tokenCount, hiddenSize, testEntries);
        addMatrixParameter("lm_head.weight", tokenCount, hiddenSize, testEntries);
        addVectorParameter("lm_head.bias", tokenCount, testEntries);
        addVectorParameter("transformer.ln_f.weight", hiddenSize, testEntries);
        addVectorParameter("transformer.ln_f.bias", hiddenSize, testEntries);

        addVectorParameter("transformer.h.0.ln_1.weight", hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.ln_1.bias", hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.q_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.k_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.v_proj.weight", hiddenSize, hiddenSize, testEntries);
        addMatrixParameter("transformer.h.0.attn.out_proj.weight", hiddenSize, hiddenSize, testEntries);

        addMatrixParameter("transformer.h.0.mlp.fc_in.weight", intermediateSize, hiddenSize, testEntries);
        addVectorParameter("transformer.h.0.mlp.fc_in.bias", intermediateSize, testEntries);
        addMatrixParameter("transformer.h.0.mlp.fc_out.weight", hiddenSize, intermediateSize, testEntries);
        addVectorParameter("transformer.h.0.mlp.fc_out.bias", hiddenSize, testEntries);

        addVectorParameter("transformer.h.0.attn.masked_bias", 1, testEntries);
        addBoolArray2DParameter("transformer.h.0.attn.bias", tokenCount, tokenCount,
                List.of(1, 1, tokenCount, tokenCount), testEntries);

        return testEntries;
    }

    private static void generate(String path, String inputSafetensors, String outputSafetensors,
                                 List<TestParameter> entries) throws IOException
    {
        // Create JSON header from the specified test entries
        var header = createSafetensorsHeader(entries);
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
            for (var entry : entries)
            {
                switch (entry.type)
                {
                    case VECTOR_PARAMETER -> {
                        // Write a vector
                        var values = parameterLoader.loadVector(reader, entry.id, entry.rows);
                        for (var i = 0; i < values.size(); i++)
                        {
                            var value = values.get(i);
                            out.writeFloat(toLittleEndian(value));
                        }
                    }
                    case MATRIX_PARAMETER ->
                    {
                        // Write a matrix
                        var values = parameterLoader.loadMatrix(reader, ParameterType.HORIZONTAL_WEIGHT, entry.id, entry.rows, entry.cols);
                        for (var row : values.getVectorArray())
                        {
                            for (var i = 0; i < row.size(); i++)
                            {
                                var value = row.get(i);
                                out.writeFloat(toLittleEndian(value));
                            }
                        }
                    }
                    case BOOL_ARRAY_2D_PARAMETER -> {
                        // Write a bool array
                        var values = parameterLoader.loadBoolArray(reader, entry.id, entry.rows, entry.cols);
                        for (var row : values)
                        {
                            for (var value: row)
                            {
                                out.writeByte(value ? 1 : 0);
                            }
                        }
                    }
                }
            }
        }
    }

    private static String createSafetensorsHeader(List<TestParameter> entries)
    {
        var startOffset = 0L;

        var metadata = new HashMap<String, String>();
        metadata.put("format", "pt");

        var safetensorsModel = new SafetensorsModel(metadata);

        for (var entry : entries)
        {
            SafetensorsDataType dataType;
            long size;
            long endOffset;

            switch (entry.type)
            {
                case VECTOR_PARAMETER ->
                {
                    dataType = SafetensorsDataType.F32;
                    size = entry.rows;
                    endOffset = startOffset + size * 4L;
                }
                case MATRIX_PARAMETER ->
                {
                    dataType = SafetensorsDataType.F32;
                    size = (long) entry.rows * entry.cols;
                    endOffset = startOffset + size * 4L;
                }
                case BOOL_ARRAY_2D_PARAMETER ->
                {
                    dataType = SafetensorsDataType.BOOL;
                    size = (long) entry.rows * entry.cols;
                    endOffset = startOffset + size;
                }
                default -> throw new RuntimeException("Unsupported parameter type");
            }

            var tensorModel = new TensorModel(dataType, entry.shape, startOffset, endOffset);
            safetensorsModel.addTensor(entry.id, tensorModel);

            startOffset = endOffset;
        }

        try
        {
            return new ObjectMapper().writeValueAsString(safetensorsModel);
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Error during safetensors header creation: " + e.getMessage());
        }
    }

    private static void addVectorParameter(String id, int rows, List<TestParameter> entries)
    {
        entries.add(new TestParameter(id, VECTOR_PARAMETER, rows, -1, List.of(rows)));
    }

    private static void addVectorParameter(String id, int rows, List<Integer> shape, List<TestParameter> entries)
    {
        entries.add(new TestParameter(id, VECTOR_PARAMETER, rows, -1, shape));
    }

    private static void addMatrixParameter(String id, int rows, int cols,
                                           List<GenerateTestSafetensors.TestParameter> entries)
    {
        entries.add(new TestParameter(id, MATRIX_PARAMETER, rows, cols, List.of(rows, cols)));
    }

    private static void addMatrixParameter(String id, int rows, int cols, List<Integer> shape,
                                           List<GenerateTestSafetensors.TestParameter> entries)
    {
        entries.add(new TestParameter(id, MATRIX_PARAMETER, rows, cols, shape));
    }

    private static void addBoolArray2DParameter(String id, int rows, int cols,
                                                List<GenerateTestSafetensors.TestParameter> entries)
    {
        entries.add(new GenerateTestSafetensors.TestParameter(id, BOOL_ARRAY_2D_PARAMETER, rows, cols,
                List.of(rows, cols)));
    }

    private static void addBoolArray2DParameter(String id, int rows, int cols, List<Integer> shape,
                                                List<GenerateTestSafetensors.TestParameter> entries)
    {
        entries.add(new GenerateTestSafetensors.TestParameter(id, BOOL_ARRAY_2D_PARAMETER, rows, cols, shape));
    }
}
