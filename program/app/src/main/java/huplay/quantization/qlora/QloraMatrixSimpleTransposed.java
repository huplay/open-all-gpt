package huplay.quantization.qlora;

import huplay.dataType.DataType;
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
    public float getValue(int rowId, int colId)
    {
        // This is the byte which holds the quantized 4-bit value
        byte value = values[rowId / 2][colId];

        // Read the 4-bit value from the lower or upper part of the byte
        byte quantizedValue = ((rowId & 1) == 0)
                ? (byte)((value & 0b11110000) >>> 4)
                : (byte) (value & 0b1111);

        // Determine which block the value is in
        int blockId = colId * blocksPerCol + (rowId / blockSize);

        // This is the de-quantization algorithm:
        return quantMap[quantizedValue] * absMax[blockId] / maxQuantMap;
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
