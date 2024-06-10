package quantization.qlora;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import app.IdentifiedException;
import config.Config;
import config.ParameterType;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import parameters.ParameterReader;
import quantization.AbstractQuantizer;
import quantization.QuantizedMatrix;

import java.util.Arrays;
import java.util.Locale;

import static math.MathUtil.MATH;
import static math.BasicMathUtility.absMax;
import static ui.TextUtil.equalsIgnoreCase;
import static java.lang.Math.abs;

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

 * @author Hunor Szegi
 */
public class QloraQuantizer extends AbstractQuantizer
{
    private static final String DEFAULT_QUANT_STATE_KEY_PREFIX = "quant_state.bitsandbytes__";
    private static final String QUANT_STATE_KEY = "quantState";
    private static final String ABS_MAX_KEY = "absMax";
    private static final String QUANT_MAP_KEY = "quantMap";
    private static final String NESTED_ABS_MAX_KEY = "nestedAbsMax";
    private static final String NESTED_QUANT_MAP_KEY = "nestedQuantMap";
    private static final String QUANTIZED_ABS_MAX_KEY = "quantizedAbsMax";

    private String quantStateKeyPrefix;

    public QloraQuantizer(Config config)
    {
        super(config);

        this.quantStateKeyPrefix = DEFAULT_QUANT_STATE_KEY_PREFIX;

        var quantizationConfig = config.getQuantizationConfig();
        var naming = quantizationConfig == null ? null : quantizationConfig.getNaming();
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
    public Matrix load(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols)
    {
        try
        {
            var variant = determineVariant(reader);
            var quantizationConfig = config.getQuantizationConfig();
            var outputFloatType = quantizationConfig.getOutputFloatType();

            // Read the QuantState, which iss a special JSON parameter, containing the settings of the quantization
            QloraQuantState quantState = readQuantState(reader, variant, parameterId);
            int blockSize = quantState.getBlockSize();

            if (!equalsIgnoreCase(variant, quantState.getQuantType()))
            {
                System.out.println("WARNING - QloRA quantization variant (" + quantState.getQuantType() + ")" +
                        " is different to the specified variant: " + variant);
            }

            // Read the quantization map for the 16 different values (which can be stored in 4 bit)
            // This is the real difference between the variants. Different variants use different set of quantiles
            float[] quantMap = reader.readFloatArray(getFinalParameterId(parameterId, QUANT_MAP_KEY), 16);

            var nestedBlockSize = quantState.getNestedBlockSize();
            if (nestedBlockSize != null)
            {
                // If the QuantState contains a nested block size value, then it is a double quantized model
                var nestedOffset = quantState.getNestedOffset();

                // Read the stored parameters for double quantized weights
                float[] nestedQuantMap = reader.readFloatArray(getFinalParameterId(parameterId, NESTED_QUANT_MAP_KEY), 256);
                float[] nestedAbsMax = reader.readFloatArray(getFinalParameterId(parameterId, NESTED_ABS_MAX_KEY), rows * cols / blockSize / nestedBlockSize);
                byte[] quantizedAbsMax = reader.readByteArray(getFinalParameterId(parameterId, QUANTIZED_ABS_MAX_KEY), cols * rows / blockSize);

                if (parameterType.isHorizontal())
                {
                    // QLoRA stores the parameters in vertical format
                    // In the case our model expects it in horizontal, transpose it...
                    var weights = reader.readByteArray2D(parameterId, cols, rows / 2); // Swapped row and col sizes
                    weights = MATH.transposeByteMatrix(weights);

                    return new QloraMatrixDQTransposed(outputFloatType, blockSize, nestedBlockSize, nestedOffset,
                            quantMap, nestedQuantMap, nestedAbsMax, quantizedAbsMax, weights);
                }
                else
                {
                    var weights = reader.readByteArray2D(parameterId, rows, cols / 2);

                    return new QloraMatrixDQ(outputFloatType, blockSize, nestedBlockSize, nestedOffset,
                            quantMap, nestedQuantMap, nestedAbsMax, quantizedAbsMax, weights);
                }
            }
            else
            {
                // Simple quantization
                float[] absMax = reader.readFloatArray(getFinalParameterId(parameterId, ABS_MAX_KEY), rows * cols / blockSize);

                if (parameterType.isHorizontal())
                {
                    // QLoRA stores the parameters in vertical format
                    // In the case our model expects it in horizontal, transpose it...
                    var weights = reader.readByteArray2D(parameterId, cols, rows / 2); // Swapped row and col sizes
                    weights = MATH.transposeByteMatrix(weights);

                    return new QloraMatrixSimpleTransposed(outputFloatType, blockSize, quantMap, absMax, weights);
                }
                else
                {
                    var weights = reader.readByteArray2D(parameterId, rows, cols / 2);

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

    private QloraQuantState readQuantState(ParameterReader reader, String variant, String parameterId) throws JsonProcessingException
    {
        parameterId = getFinalParameterId(parameterId, QUANT_STATE_KEY).replace("{variant}", variant.toLowerCase(Locale.ROOT));

        var quantStateJson = reader.readString(parameterId);
        return new ObjectMapper().readValue(quantStateJson, QloraQuantState.class);
    }

    @Override
    public QuantizedMatrix quantize(ParameterType parameterType, Matrix matrix)
    {
        var quantMap = new float[] {
                -1.0f,
                -0.6961928009986877f,
                -0.5250730514526367f,
                -0.39491748809814453f,
                -0.28444138169288635f,
                -0.18477343022823334f,
                -0.09105003625154495f,
                0.0f,
                0.07958029955625534f,
                0.16093020141124725f,
                0.24611230194568634f,
                0.33791524171829224f,
                0.44070982933044434f,
                0.5626170039176941f,
                0.7229568362236023f,
                1.0f};

        var rows = matrix.getRowCount();
        var cols = matrix.getColCount();
        var blockSize = 64;
        var blocksPerRow = cols / blockSize;
        var maxQuant = absMax(quantMap);

        // TODO: This logic works only if the cols can be divided by the blockSize
        // If we want to make it more general, we should flatten the input matrix,
        // and at the end split the result into rows.
        // OR: refactor the Qlora matrices using a flatten storage

        var absMax = new float[rows * cols / blockSize];
        var values = new byte[rows][cols / 2];

        var blockId = 0;
        for (var rowId = 0; rowId < matrix.getRowCount(); rowId++)
        {
            var row = matrix.row(rowId).getValues();

            for (var blockIndex = 0; blockIndex < blocksPerRow; blockIndex++)
            {
                var startIndex = blockIndex * blockSize;
                float[] block = Arrays.copyOfRange(row, startIndex, startIndex + blockSize);

                absMax[blockId] = absMax(block);
                var quantConstant = maxQuant / absMax[blockId];

                for (int i = 0; i < blockSize / 2; i++)
                {
                    var lowerValue = findNearest(quantMap, block[i * 2] * quantConstant);
                    var upperValue = findNearest(quantMap, block[i * 2 + 1] * quantConstant);

                    values[rowId][i + blockIndex * blockSize / 2] =
                            (byte)(((lowerValue & 0b1111) << 4) + (upperValue & 0b1111));
                }

                blockId++;
            }
        }

        return new QloraMatrixSimple(DataType.FLOAT_32, blockSize, quantMap, absMax, values);
    }

    private int findNearest(float[] quantMap,  float value)
    {
        var nearest = 0;
        var diff = abs(value - quantMap[0]);

        for (var i = 1; i < quantMap.length; i++)
        {
            var newDiff = abs(value - quantMap[i]);
            if (newDiff < diff)
            {
                nearest = i;
                diff = newDiff;
            }
        }

        return nearest;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int size)
    {
        try
        {
            var variant = determineVariant(reader);
            QloraQuantState quantState = readQuantState(reader, variant, parameterId);
            int blockSize = quantState.getBlockSize();

            if (quantState.getNestedBlockSize() == null)
            {
                // Simple matrix
                return 12L + // int blockSize, blocksPerRow; float maxQuantMap
                        16L * 4L + // float[] quantMap
                        (size / blockSize * 4L) + // float[] absMax
                        (size / 2); // byte[][] values
            }
            else
            {
                // Double quantized matrix
                return 24L + // int blockSize, nestedBlockSize, blocksPerRow; float nestedOffset, maxQuantMap, maxNestedQuantMap
                        16L * 4L + // float[] quantMap
                        256L * 4L + // float[] nestedQuantMap
                        (size / blockSize / quantState.getNestedBlockSize() * 4L) + // float[] nestedAbsMax
                        (size / blockSize / 2) + // float[] quantizedAbsMax
                        (size / 2); // byte[][] values
            }
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Error at calculating byte size. Id: " + parameterId, e);
        }
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int rows, int cols)
    {
        return calculateByteSize(reader, parameterId, rows * cols);
    }
}