package huplay.quantization.qlora;

import huplay.IdentifiedException;
import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;

import java.util.Locale;

public class QloraFp4Matrix implements Matrix
{
    private final int blockSize;
    private final float[] quantMap;
    private final float[] absMax;
    private final byte[][] values;
    private final String outputFloatType;
    private final int blocksPerRow;

    public QloraFp4Matrix(int blockSize, float[] quantMap, float[] absMax, byte[][] values, String outputFloatType)
    {
        this.blockSize = blockSize;
        this.quantMap = quantMap;
        this.absMax = absMax;
        this.values = values;
        this.outputFloatType = outputFloatType;
        this.blocksPerRow = getColCount() / blockSize;
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
        var value = values[row][Math.floorDiv(col, 2)];

        var quantizedValue = (col % 2 == 0)
                ? (byte)((value & 0b11110000) >>> 4)
                : (byte) (value & 0b1111);

        var blockId = row * blocksPerRow + Math.floorDiv(col, blockSize);

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
        return values.length;
    }

    @Override
    public int getColCount()
    {
        return values[0].length * 2;
    }

    @Override
    public DataType getInternalFloatType()
    {
        return DataType.valueOf(outputFloatType.toUpperCase(Locale.ROOT));
    }
}
