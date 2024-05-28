package huplay.quantization.qlora;

import huplay.IdentifiedException;
import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;

import java.util.Locale;

import static huplay.math.TypeConversionUtility.*;

public class QloraFp4TransposedMatrix implements Matrix
{
    private final String variant;
    private final int blockSize;
    private final float[] quantMap;
    private final float[] absMax;
    private final byte[][] values;
    private final String outputFloatType;
    private final int blocksPerCol;

    public QloraFp4TransposedMatrix(String variant, int blockSize, float[] quantMap, float[] absMax, byte[][] values,
                                    String outputFloatType)
    {
        this.variant = variant;
        this.blockSize = blockSize;
        this.quantMap = quantMap;
        this.absMax = absMax;
        this.values = values;
        this.outputFloatType = outputFloatType;
        this.blocksPerCol = getRowCount() / blockSize;
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
    public Vector getRow(int row)
    {
        var vector = Vector.emptyVector(getInternalFloatType(), getColCount());

        for (int i = 0; i < getColCount(); i++)
        {
            vector.set(i, getValue(row, i));
        }

        return vector;
    }

    @Override
    public void setRow(int row, Vector vector)
    {
        throw new IdentifiedException("Unsupported operation: setVector on quantized matrix");
    }

    @Override
    public float getValue(int row, int col)
    {
        var value = values[Math.floorDiv(row, 2)][col];

        var quantizedValue = (row % 2 == 0)
                ? getLower4bitsFromUnsignedByte(value)
                : getUpper4bitsFromUnsignedByte(value);

        var blockId = col * blocksPerCol + Math.floorDiv(row, blockSize);

        return quantMap[quantizedValue] * absMax[blockId];
    }

    @Override
    public void setValue(int row, int col, float value)
    {
        throw new IdentifiedException("Unsupported operation: setValue on quantized matrix");
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

    @Override
    public DataType getInternalFloatType()
    {
        return DataType.valueOf(outputFloatType.toUpperCase(Locale.ROOT));
    }
}
