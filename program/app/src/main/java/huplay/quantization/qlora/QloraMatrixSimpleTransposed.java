package huplay.quantization.qlora;

import huplay.dataType.DataType;
import huplay.dataType.vector.Vector;
import huplay.quantization.QuantizedMatrix;

import static huplay.math.BasicMathUtility.absMax;

public class QloraMatrixSimpleTransposed extends QuantizedMatrix
{
    private final int blockSize;

    private final float[] quantMap;
    private final float[] absMax;
    private final byte[][] values;

    private final int blocksPerCol;
    private final float maxQuantMap;

    public QloraMatrixSimpleTransposed(DataType outputFloatType, int blockSize, float[] quantMap, float[] absMax,
                                       byte[][] values)
    {
        super(outputFloatType);
        this.blockSize = blockSize;

        this.quantMap = quantMap;
        this.absMax = absMax;
        this.values = values;

        this.blocksPerCol = getRowCount() / blockSize;
        this.maxQuantMap = absMax(quantMap);
    }

    @Override
    public float getValue(int row, int col)
    {
        // This is the byte which holds the quantized 4-bit value
        byte value = values[row / 2][col];

        // Read the 4-bit value from the lower or upper part of the byte
        byte quantizedValue = ((row & 1) == 0)
                ? (byte)((value & 0b11110000) >>> 4)
                : (byte) (value & 0b1111);

        // Determine which block the value is in
        int blockId = col * blocksPerCol + (row / blockSize);

        // This is the de-quantization algorithm:
        return quantMap[quantizedValue] * absMax[blockId] / maxQuantMap;
    }

    @Override
    public Vector[] getVectorArray()
    {
        var vectorArray = new Vector[values.length];

        for (int i = 0; i < values.length; i++)
        {
            vectorArray[i] = getRow(i);
        }

        return vectorArray;
    }

    @Override
    public Vector getRow(int row) // TODO: Should we transpose it?
    {
        var vector = Vector.emptyVector(getInternalFloatType(), getColCount());

        for (int i = 0; i < getColCount(); i++)
        {
            vector.set(i, getValue(row, i));
        }

        return vector;
    }

    @Override
    public int getRowCount()
    {
        return values.length * 2;
    }

    @Override
    public int getColCount()
    {
        return values[0].length;
    }
}
