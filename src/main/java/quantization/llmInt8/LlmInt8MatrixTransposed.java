package quantization.llmInt8;

import math.dataType.DataType;
import quantization.QuantizedMatrix;

/**
 * Matrix which holds the LLM.int8() quantized values in transposed orientation,
 * and processes the de-quantization when the getValue() is called
 */
public class LlmInt8MatrixTransposed extends QuantizedMatrix
{
    // Worst naming ever: "SCB" means "the quantization state that belongs to B":
    // https://github.com/TimDettmers/bitsandbytes/issues/540
    private final float[] scb;
    private final byte[][] quantizedValues;

    public LlmInt8MatrixTransposed(DataType outputFloatType, float[] scb, byte[][] quantizedValues)
    {
        super(outputFloatType);
        this.scb = scb;
        this.quantizedValues = quantizedValues;
    }

    @Override
    public float getValue(int rowId, int colId)
    {
        // This is the de-quantization algorithm:
        // The quantized values are stored in transposed orientation, so the rowId and colId are swapped.
        return scb[colId] * quantizedValues[colId][rowId] / 127;
    }

    @Override
    public int getRowCount()
    {
        // The quantized values are stored in transposed orientation, so it returns the number of columns as rowCount
        return quantizedValues[0].length;
    }

    @Override
    public int getColCount()
    {
        // The quantized values are stored in transposed orientation, so it returns the number of rows as colCount
        return quantizedValues.length;
    }
}
