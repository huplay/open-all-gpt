package huplay.quantization.qlora;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.ParameterReader;
import huplay.parameters.StandardParameterLoader;

import java.util.HashMap;
import java.util.Map;

import static huplay.MathUtilProvider.MATH;
import static huplay.ui.TextUtil.equalsIgnoreCase;

/**
 * Parameter reader for QLoRA quantization (QLoRA de-quantization)

 * QLoRA (Quantized Low Rank Adapters) (University of Washington)
 * Publication (23 May 2023): https://arxiv.org/abs/2305.14314

 * QLoRA has multiple variants:
 * - int4
 * - fp4 (pure float 4)
 * - nf4 (normalized float 4)
 * - af4 (abnormal float 4) (https://arxiv.org/abs/2306.06965)
 * with or without double quantization (DQ, where the absMax is quantized again)
 * (nf4 with DQ is the recommended variant.)

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
    private final Config config;
    private final Map<String, String> defaultNamingMap = new HashMap<>();

    public QloraParameterLoader(Config quantizationConfig)
    {
        this.config = quantizationConfig;

        defaultNamingMap.put("quantStateFP4", "{name}.quant_state.bitsandbytes__fp4");
    }

    @Override
    public Matrix readMatrix(ParameterReader reader, ParameterType parameterType, String id, int rows, int cols)
    {
        try
        {
            var quantizationConfig = config.getQuantizationConfig();
            var outputFloatType = quantizationConfig.getOutputFloatType();
            var variant = quantizationConfig.getVariant();
            if (variant == null) variant = "fp4";

            QloraQuantState quantState = readQuantState(reader, variant, id);
            int blockSize = quantState.getBlockSize();

            if (!equalsIgnoreCase(variant, quantState.getQuantType()))
            {
                System.out.println("WARNING - QloRA quantization variant (" + quantState.getQuantType() + ")" +
                        " is different to the specified variant: " + variant);
            }

            // Read and unpack the quantized collections
            float[] absMax = readAbsMax(reader, id, rows * cols / blockSize);
            float[] quantMap = readQuantMap(reader, id);

            if (quantizationConfig.getTransposeMatrix())
            {
                var weights = readQuantizedWeights(reader, id, cols, rows);
                weights  = MATH.transposeByteMatrix(weights);
                return new QloraFp4TransposedMatrix(variant, blockSize, quantMap, absMax, weights, outputFloatType);
            }
            else
            {
                var weights = readQuantizedWeights(reader, id, rows, cols);
                return new QloraFp4Matrix(variant, blockSize, quantMap, absMax, weights, outputFloatType);
            }
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Error during reading data for QLoRA quantization", e);
        }
    }

    private QloraQuantState readQuantState(ParameterReader reader, String variant, String id) throws JsonProcessingException
    {
        var namingMap = config.getQuantizationConfig().getNaming();
        if (namingMap == null) namingMap = defaultNamingMap;

        var quantStateName = namingMap.get("quantStateFP4");

        if (variant == null || equalsIgnoreCase(variant, "fp4"))
        {
            quantStateName = quantStateName.replace("{name}", id);
            var quantStateJson = reader.readString(quantStateName);
            return new ObjectMapper().readValue(quantStateJson, QloraQuantState.class);
        }
        else
        {
            // TODO
            return null;
        }
    }

    // TODO: support naming
    private float[] readAbsMax(ParameterReader reader, String id, int rows)
    {
        return reader.readFloatArray(id + ".absmax", rows);
    }

    private float[] readQuantMap(ParameterReader reader, String id)
    {
        return reader.readFloatArray(id + ".quant_map", 16);
    }

    private byte[][] readQuantizedWeights(ParameterReader reader, String id, int rows, int cols)
    {
        return reader.readByteArray2D(id, rows, cols / 2);
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String id, int size)
    {
        // TODO: Calculate
        return 0;
    }
}