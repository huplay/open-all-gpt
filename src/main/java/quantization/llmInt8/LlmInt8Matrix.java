package quantization.llmInt8;

import math.dataType.DataType;
import quantization.QuantizedMatrix;

/**
 * Matrix which holds the LLM.int8() quantized values,
 * and processes the de-quantization when the getValue() is called
 */
public class LlmInt8Matrix extends QuantizedMatrix
{
    // Worst naming ever: "SCB" means "the quantization state that belongs to B":
    // https://github.com/TimDettmers/bitsandbytes/issues/540
    private final float[] scb;
    private final byte[][] quantizedValues;

    public LlmInt8Matrix(DataType outputFloatType, float[] scb, byte[][] quantizedValues)
    {
        super(outputFloatType);
        this.scb = scb;
        this.quantizedValues = quantizedValues;
    }

    @Override
    public float getValue(int rowId, int colId)
    {
        // This is the de-quantization algorithm:
        return scb[colId] * quantizedValues[rowId][colId] / 127;
    }

    @Override
    public int getRowCount()
    {
        return quantizedValues.length;
    }

    @Override
    public int getColCount()
    {
        return quantizedValues[0].length;
    }
}
