package huplay.math;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TypeConversionUtility
{
    /**
     * Unpacks an int (32 bit) value into 8 signed 4-bit int values,
     * and appends it into an array at the given offset.
     */
    public static void unpack4bitsFromInt32(int value, int[] array, int offset)
    {
        array[    offset] = ((value & 0b1111_0000_0000_0000_0000_0000_0000_0000) >>> 28) - 8;
        array[1 + offset] = ((value & 0b0000_1111_0000_0000_0000_0000_0000_0000) >>> 24) - 8;
        array[2 + offset] = ((value & 0b0000_0000_1111_0000_0000_0000_0000_0000) >>> 20) - 8;
        array[3 + offset] = ((value & 0b0000_0000_0000_1111_0000_0000_0000_0000) >>> 16) - 8;
        array[4 + offset] = ((value & 0b0000_0000_0000_0000_1111_0000_0000_0000) >>> 12) - 8;
        array[5 + offset] = ((value & 0b0000_0000_0000_0000_0000_1111_0000_0000) >>> 8) - 8;
        array[6 + offset] = ((value & 0b0000_0000_0000_0000_0000_0000_1111_0000) >>> 4) - 8;
        array[7 + offset] = (value & 0b0000_0000_0000_0000_0000_0000_0000_1111) - 8;
    }

    public static int[][] unpack4bitsFromIntMatrixByRow(int[][] matrix)
    {
        int[][] result = new int[matrix.length][matrix[0].length * 8];

        int i = 0;
        for (int[] row : matrix)
        {
            var offset = 0;
            for (int value : row)
            {
                unpack4bitsFromInt32(value, result[i], offset);
                offset += 8;
            }
            i++;
        }

        return result;
    }

    public static int[][] unpackIntMatrixByCol(int[][] matrix, int n, int size)
    {
        int[][] result = new int[matrix.length * n][matrix[0].length];

        int rowOffset = 0;
        for (int[] ints : matrix)
        {
            for (var col = 0; col < matrix[0].length; col++)
            {
                int[] array = new int[n];
                unpack4bitsFromInt32(ints[col], array, 0);

                for (var i = 0; i < array.length; i++)
                {
                    result[rowOffset + i][col] = array[i];
                }
            }

            rowOffset += n;
        }

        return result;
    }

    public static long toLittleEndian(long value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putLong(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getLong(0);
    }

    public static float toLittleEndian(float value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putFloat(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getFloat(0);
    }
}
