package huplay.quantization.qlora;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.ParameterReader;
import huplay.parameters.StandardParameterLoader;

import java.util.Locale;

import static huplay.MathUtilProvider.MATH;
import static huplay.ui.TextUtil.equalsIgnoreCase;

/**
 * Parameter reader for QLoRA quantization (QLoRA de-quantization)

 * QLoRA (Quantized Low Rank Adapters) (University of Washington)
 * Publication (23 May 2023): https://arxiv.org/abs/2305.14314

 * QLoRA has multiple variants (int4, fp4, nf4, af4), with or without double quantization
 * The used variant is determined automatically

 * The main author of LLM.int8() and QLoRA is the same person: Tim Dettmers, who is leader of bitsandbytes.
 * Hugging Face has a collaboration with bitsandbytes, so the bitsandbytes quantization implementations
 * are included into the Hugging Face's Transformers framework.
 * Later bitsandbytes supported other quantization methods, like GPTQ (AutoGPTQ), AWQ (AutoAWQ), etc.
 * But the first two, own methods are branded as "bitsandbytes".
 * (If it is 8-bit, that is the LLM.int8(), if it is 4-bit, that is QLoRA.)
 * https://huggingface.co/blog/4bit-transformers-bitsandbytes

 * Other groups also created quantization frameworks, which can be based on 4-bit "bitsandbytes" quantization,
 * which is in practice QLoRA (for example PruneAI).

 * (Only the matrix quantization is supported, so the vector reading is inherited from the standard parameter loader.)
 * @author Hunor Szegi
 */
public class QloraParameterLoader extends StandardParameterLoader
{
    private static final String DEFAULT_QUANT_STATE_KEY_PREFIX = "quant_state.bitsandbytes__";
    private static final String QUANT_STATE_KEY = "quantState";
    private static final String ABS_MAX_KEY = "absMax";
    private static final String QUANT_MAP_KEY = "quantMap";
    private static final String NESTED_ABS_MAX_KEY = "nestedAbsMax";
    private static final String NESTED_QUANT_MAP_KEY = "nestedQuantMap";
    private static final String QUANTIZED_ABS_MAX_KEY = "quantizedAbsMax";

    private String quantStateKeyPrefix;

    public QloraParameterLoader(Config config)
    {
        super(config);

        this.quantStateKeyPrefix = DEFAULT_QUANT_STATE_KEY_PREFIX;

        var naming = config.getQuantizationConfig().getNaming();
        if (naming != null && naming.get("QUANT_STATE_KEY_PREFIX") != null)
        {
            this.quantStateKeyPrefix = naming.get("QUANT_STATE_KEY_PREFIX");
        }

        addDefaultNaming(QUANT_STATE_KEY, "{name}." + quantStateKeyPrefix + "{variant}");
        addDefaultNaming(ABS_MAX_KEY, "{name}.absmax");
        addDefaultNaming(QUANT_MAP_KEY, "{name}.quant_map");
        addDefaultNaming(NESTED_ABS_MAX_KEY, "{name}.nested_absmax");
        addDefaultNaming(NESTED_QUANT_MAP_KEY, "{name}.nested_quant_map");
        addDefaultNaming(QUANTIZED_ABS_MAX_KEY, "{name}.absmax");
    }

    @Override
    public Matrix readMatrix(ParameterReader reader, ParameterType parameterType, String id, int rows, int cols)
    {
        try
        {
            var variant = determineVariant(reader);
            var quantizationConfig = config.getQuantizationConfig();
            var outputFloatType = quantizationConfig.getOutputFloatType();

            // Read the QuantState, which iss a special JSON parameter, containing the settings of the quantization
            QloraQuantState quantState = readQuantState(reader, variant, id);
            int blockSize = quantState.getBlockSize();

            if (!equalsIgnoreCase(variant, quantState.getQuantType()))
            {
                System.out.println("WARNING - QloRA quantization variant (" + quantState.getQuantType() + ")" +
                        " is different to the specified variant: " + variant);
            }

            // Read the quantization map for the 16 different values (which can be stored in 4 bit)
            // This is the real difference between the variants. Different variants use different set of quantiles
            float[] quantMap = readQuantMap(reader, id, 16);

            var nestedBlockSize = quantState.getNestedBlockSize();
            if (nestedBlockSize != null)
            {
                // If the QuantState contains a nested block size value, then it is a double quantized model
                var nestedOffset = quantState.getNestedOffset();

                // Read the stored parameters for double quantized weights
                float[] nestedQuantMap = readNestedQuantMap(reader, id, 256);
                float[] nestedAbsMax = readNestedAbsMax(reader, id, rows * cols / blockSize / nestedBlockSize);
                byte[] quantizedAbsMax = readQuantizedAbsMax(reader, id, cols * rows / blockSize);

                if (quantizationConfig.getTransposeMatrix())
                {
                    var weights = readQuantizedWeights(reader, id, cols, rows); // Swapped row and col sizes
                    weights = MATH.transposeByteMatrix(weights);

                    return new QloraMatrixDQTransposed(outputFloatType, blockSize, nestedBlockSize, nestedOffset,
                            quantMap, nestedQuantMap, nestedAbsMax, quantizedAbsMax, weights);
                }
                else
                {
                    var weights = readQuantizedWeights(reader, id, rows, cols);

                    return new QloraMatrixDQ(outputFloatType, blockSize, nestedBlockSize, nestedOffset,
                            quantMap, nestedQuantMap, nestedAbsMax, quantizedAbsMax, weights);
                }
            }
            else
            {
                // Simple quantization
                float[] absMax = readAbsMax(reader, id, rows * cols / blockSize);

                if (quantizationConfig.getTransposeMatrix())
                {
                    var weights = readQuantizedWeights(reader, id, cols, rows); // Swapped row and col sizes
                    weights = MATH.transposeByteMatrix(weights);

                    return new QloraMatrixSimpleTransposed(outputFloatType, blockSize, quantMap, absMax, weights);
                }
                else
                {
                    var weights = readQuantizedWeights(reader, id, rows, cols);

                    return new QloraMatrixSimple(outputFloatType, blockSize, quantMap, absMax, weights);
                }
            }
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Error during reading data for QLoRA quantization", e);
        }
    }

    private String determineVariant(ParameterReader reader)
    {
        // At QLoRA there is a special parameter, which contains the settings of the quantization in JSON format
        // The name of the parameter ends with "quant_state.bitsandbytes__{variant}"
        // For example: "transformer.h.0.attn.c_attn.weight.quant_state.bitsandbytes__nf4"
        // Here we determine the used variant based on the "nf4" (or "fp4, etc.) suffix.

        for (var key : reader.getParameterHeaders().keySet())
        {
            if (key.contains(quantStateKeyPrefix))
            {
                var index = key.indexOf(quantStateKeyPrefix);
                return key.substring(index + quantStateKeyPrefix.length());
            }
        }

        return "nf4";
    }

    private QloraQuantState readQuantState(ParameterReader reader, String variant, String id) throws JsonProcessingException
    {
        id = getFinalId(id, QUANT_STATE_KEY).replace("{variant}", variant.toLowerCase(Locale.ROOT));

        var quantStateJson = reader.readString(id);
        return new ObjectMapper().readValue(quantStateJson, QloraQuantState.class);
    }

    private float[] readQuantMap(ParameterReader reader, String id, int size)
    {
        id = getFinalId(id, QUANT_MAP_KEY);
        return reader.readFloatArray(id, size);
    }

    private float[] readAbsMax(ParameterReader reader, String id, int size)
    {
        id = getFinalId(id, ABS_MAX_KEY);
        return reader.readFloatArray(id, size);
    }

    private byte[][] readQuantizedWeights(ParameterReader reader, String id, int rows, int cols)
    {
        return reader.readByteArray2D(id, rows, cols / 2);
    }

    private float[] readNestedQuantMap(ParameterReader reader, String id, int size)
    {
        id = getFinalId(id, NESTED_QUANT_MAP_KEY);
        return reader.readFloatArray(id, size);
    }

    private float[] readNestedAbsMax(ParameterReader reader, String id, int size)
    {
        id = getFinalId(id, NESTED_ABS_MAX_KEY);
        return reader.readFloatArray(id, size);
    }

    private byte[] readQuantizedAbsMax(ParameterReader reader, String id, int size)
    {
        id = getFinalId(id, QUANTIZED_ABS_MAX_KEY);
        return reader.readByteArray(id, size);
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String id, int size)
    {
        // TODO: Calculate
        return 0;
    }
}