package huplay.quantization.llmInt8;

import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.ParameterReader;
import huplay.parameters.StandardParameterLoader;

import static huplay.MathUtilProvider.MATH;

/**
 * Parameter reader for LLM.int8() quantization (LLM.int8() de-quantization)

 * LLM.int8() publication (15 Aug 2022): https://arxiv.org/abs/2208.07339
 
 * The main author of LLM.int8() and QLoRA is the same person: Tim Dettmers, who is leader of bitsandbytes.
 * Hugging Face has a collaboration with bitsandbytes, so the bitsandbytes quantization implementations
 * are included into the Hugging Face's Transformers framework.
 * Later bitsandbytes supported other quantization methods, like GPTQ (AutoGPTQ), AWQ (AutoAWQ), etc.
 * But the first two, own methods are branded as "bitsandbytes".
 * (If it is 8-bit, that is the LLM.int8(), if it is 4-bit, that is QLoRA.)
 * https://huggingface.co/blog/4bit-transformers-bitsandbytes

 * (Only the matrix quantization is supported, so the vector reading is inherited from the standard parameter loader.)
 * @author Hunor Szegi
 */
public class LlmInt8ParameterLoader extends StandardParameterLoader
{
    private static final String SCB_KEY = "scb";
    private static final String WEIGHTS_KEY = "weights";

    public LlmInt8ParameterLoader(Config config)
    {
        super(config);

        addDefaultNaming(SCB_KEY, "{name-1}.SCB");
        addDefaultNaming(WEIGHTS_KEY, "{name}");
    }

    @Override
    public Matrix readMatrix(ParameterReader reader, ParameterType parameterType, String id, int rows, int cols)
    {
        var quantizationConfig = config.getQuantizationConfig();
        var outputFloatType = quantizationConfig.getOutputFloatType();

        float[] scb = readSCB(reader, id, cols);
        byte[][] weights = readQuantizedWeights(reader, id, cols, rows);
        weights = MATH.transposeByteMatrix(weights);

        return new LlmInt8Matrix(outputFloatType, scb, weights);
    }

    private float[] readSCB(ParameterReader reader, String id, int size)
    {
        // Worst naming ever: "SCB" means the quantization state that belongs to B:
        // https://github.com/TimDettmers/bitsandbytes/issues/540
        id = getFinalId(id, SCB_KEY);
        return reader.readFloatArray(id, size);
    }

    private byte[][] readQuantizedWeights(ParameterReader reader, String id, int rows, int cols)
    {
        return reader.readByteArray2D(id, rows, cols);
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String id, int size)
    {
        // TODO: Calculate
        return size;
    }
}