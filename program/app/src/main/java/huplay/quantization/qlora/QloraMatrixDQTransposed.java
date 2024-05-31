package huplay.quantization.qlora;

import huplay.dataType.DataType;
import huplay.dataType.vector.Vector;
import huplay.quantization.QuantizedMatrix;

import static huplay.math.BasicMathUtility.absMax;

public class QloraMatrixDQTransposed extends QuantizedMatrix
{
    private final int blockSize;
    private final int nestedBlockSize;
    private final float nestedOffset;

    private final float[] quantMap;
    private final float[] nestedQuantMap;
    private final float[] nestedAbsMax;
    private final byte[] quantizedAbsMax;
    private final byte[][] values;

    private final int blocksPerCol;
    private final float maxQuantMap;
    private final float maxNestedQuantMap;

    public QloraMatrixDQTransposed(DataType outputFloatType, int blockSize, int nestedBlockSize, float nestedOffset,
                                   float[] quantMap, float[] nestedQuantMap, float[] nestedAbsMax,
                                   byte[] quantizedAbsMax, byte[][] values)
    {
        super(outputFloatType);
        this.blockSize = blockSize;
        this.nestedBlockSize = nestedBlockSize;
        this.nestedOffset = nestedOffset;

        this.quantMap = quantMap;
        this.nestedQuantMap = nestedQuantMap;
        this.nestedAbsMax = nestedAbsMax;
        this.quantizedAbsMax = quantizedAbsMax;
        this.values = values;

        this.blocksPerCol = getRowCount() / blockSize;
        this.maxQuantMap = absMax(quantMap);
        this.maxNestedQuantMap = absMax(nestedQuantMap);
    }

    @Override
    public float getValue(int row, int col)
    {
        // This is the byte which holds the quantized 4-bit value
        byte value = values[row / 2][col];

        // Read the 4-bit value from the lower or upper part of the byte
        byte quantizedValue = ((row & 1) == 0)
                ? (byte) ((value & 0b11110000) >>> 4)
                : (byte) (value & 0b1111);

        // Determine which block the value is in
        int blockId = col * blocksPerCol + (row / blockSize);

        // We have double quantization, so the absMax is also quantized. De-quantize it!
        float absMax = deQuantizeAbsMax(blockId);

        // This is the de-quantization algorithm:
        return quantMap[quantizedValue] * absMax / maxQuantMap;
    }

    private float deQuantizeAbsMax(int pos)
    {
        // The nested quantization uses 8-bit quantized values pointing to one of the 256 quantiles

        // This is the byte which holds the quantized 8-bit value of the absMax
        int quantizedValue = quantizedAbsMax[pos] + 128; // To convert from unsigned byte to signed 128 is added

        // Determine which nested block the value in
        int nestedBlockId = pos / nestedBlockSize;

        // This is the de-quantization algorithm for the nested case:
        return nestedQuantMap[quantizedValue] * nestedAbsMax[nestedBlockId] / maxNestedQuantMap + nestedOffset;
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
