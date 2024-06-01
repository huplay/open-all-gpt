package huplay.quantization.llmInt8;

import huplay.dataType.DataType;
import huplay.dataType.vector.Vector;
import huplay.quantization.QuantizedMatrix;

public class LlmInt8Matrix extends QuantizedMatrix
{
    private final float[] scales;
    private final byte[][] quantizedValues;

    public LlmInt8Matrix(DataType outputFloatType, float[] scales, byte[][] quantizedValues)
    {
        super(outputFloatType);
        this.scales = scales;
        this.quantizedValues = quantizedValues;
    }

    @Override
    public float getValue(int row, int col)
    {
        // This is the de-quantization algorithm:
        return scales[col] * quantizedValues[row][col] / 127;
    }

    @Override
    public Vector[] getVectorArray()
    {
        var vectorArray = new Vector[quantizedValues.length];

        for (int i = 0; i < quantizedValues.length; i++)
        {
            vectorArray[i] = getRow(i);
        }

        return vectorArray;
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
