package huplay.quantization.qlora;

import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.quantization.QuantizedMatrix;

import java.util.Arrays;

import static huplay.math.BasicMathUtility.absMax;
import static java.lang.Math.abs;

public class QloraQuantizer
{
    public static QuantizedMatrix quantize(Matrix matrix)
    {
        var quantMap = new float[] {
                -1.0f,
                -0.6961928009986877f,
                -0.5250730514526367f,
                -0.39491748809814453f,
                -0.28444138169288635f,
                -0.18477343022823334f,
                -0.09105003625154495f,
                0.0f,
                0.07958029955625534f,
                0.16093020141124725f,
                0.24611230194568634f,
                0.33791524171829224f,
                0.44070982933044434f,
                0.5626170039176941f,
                0.7229568362236023f,
                1.0f};

        var rows = matrix.getRowCount();
        var cols = matrix.getColCount();
        var blockSize = 64;
        var blocksPerRow = cols / blockSize;
        var maxQuant = absMax(quantMap);

        // TODO: This logic works only if the cols can be divided by the blockSize
        // If we want to make it more general, we should flatten the input matrix,
        // and at the end split the result into rows.
        // OR: refactor the Qlora matrices using a flatten storage

        var absMax = new float[rows * cols / blockSize];
        var values = new byte[rows][cols / 2];

        var blockId = 0;
        for (var rowId = 0; rowId < matrix.getRowCount(); rowId++)
        {
            var row = matrix.getRow(rowId).getValues();

            for (var blockIndex = 0; blockIndex < blocksPerRow; blockIndex++)
            {
                var startIndex = blockIndex * blockSize;
                float[] block = Arrays.copyOfRange(row, startIndex, startIndex + blockSize);

                absMax[blockId] = absMax(block);
                var quantConstant = maxQuant / absMax[blockId];

                for (int i = 0; i < blockSize / 2; i++)
                {
                    var lowerValue = findNearest(quantMap, block[i * 2] * quantConstant);
                    var upperValue = findNearest(quantMap, block[i * 2 + 1] * quantConstant);

                    values[rowId][i + blockIndex * blockSize / 2] = pack(lowerValue, upperValue);
                }

                blockId++;
            }
        }

        return new QloraMatrixSimple(DataType.FLOAT_32, blockSize, quantMap, absMax, values);
    }

    private static int findNearest(float[] quantMap,  float value)
    {
        var nearest = 0;
        var diff = abs(value - quantMap[0]);

        for (var i = 1; i < quantMap.length; i++)
        {
            var newDiff = abs(value - quantMap[i]);
            if (newDiff < diff)
            {
                nearest = i;
                diff = newDiff;
            }
        }

        return nearest;
    }

    private static byte pack(int lowerValue, int upperValue)
    {
        return (byte)(((lowerValue & 0b1111) << 4) + (upperValue & 0b1111));
    }
}
