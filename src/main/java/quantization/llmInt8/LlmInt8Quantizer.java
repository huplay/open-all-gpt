package quantization.llmInt8;

import config.Config;
import config.ParameterType;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import parameters.ParameterReader;
import quantization.AbstractQuantizer;
import quantization.QuantizedMatrix;

import static math.MathUtil.MATH;
import static math.BasicMathUtility.absMax;

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

 * @author Hunor Szegi
 */
public class LlmInt8Quantizer extends AbstractQuantizer
{
    private static final String SCB_KEY = "scb";
    private static final String WEIGHTS_KEY = "weights";

    public LlmInt8Quantizer(Config config)
    {
        super(config);
        addDefaultNaming(SCB_KEY, "{name-1}.SCB");
        addDefaultNaming(WEIGHTS_KEY, "{name}");
    }

    @Override
    public Matrix load(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols)
    {
        float[] scb = reader.readFloatArray(getFinalParameterId(parameterId, SCB_KEY), cols);

        byte[][] weights = reader.readByteArray2D(getFinalParameterId(parameterId, WEIGHTS_KEY), cols, rows);

        if (parameterType.isHorizontal())
        {
            // LLM.int8() stores the parameters in vertical format
            // In the case our model expects it in horizontal, transpose it...
            weights = MATH.transposeByteMatrix(weights);
        }

        var outputFloatType = config.getQuantizationConfig().getOutputFloatType();
        return new LlmInt8Matrix(outputFloatType, scb, weights);
    }

    @Override
    public QuantizedMatrix quantize(ParameterType parameterType, Matrix matrix)
    {
        if (parameterType.isHorizontal())
        {
            // LLM.int8() expects the parameters in vertical orientation
            // In the case our model stores it in horizontal, transpose it, and later transpose the result again
            //
            // Other options:
            //   - implement the quantization for transpose orientation as well
            //   - transpose the values only once, and use a transposed matrix (It is implemented, but not used)
            matrix = MATH.transposeMatrix(matrix);
        }

        var rows = matrix.getRowCount();
        var cols = matrix.getColCount();

        var scb = new float[rows];
        var values = new byte[rows][cols];

        for (var rowId = 0; rowId < matrix.getRowCount(); rowId++)
        {
            var row = matrix.row(rowId).getValues();

            scb[rowId] = absMax(row);
            var quantConstant = 127 / scb[rowId];

            for (int colId = 0; colId < cols; colId++)
            {
                values[rowId][colId] = (byte)(matrix.getValue(rowId, colId) * quantConstant);
            }
        }

        if (parameterType.isHorizontal())
        {
            // If our model expects the parameter in horizontal orientation, transpose back the result
            values = MATH.transposeByteMatrix(values);
        }

        return new LlmInt8Matrix(DataType.FLOAT_16, scb, values);
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int size)
    {
        return size + 4;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int rows, int cols)
    {
        return rows * (cols + 4L);
    }
}
