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
        array[offset    ] = ((value & 0b1111_0000_0000_0000_0000_0000_0000_0000) >>> 28) - 8;
        array[offset + 1] = ((value & 0b0000_1111_0000_0000_0000_0000_0000_0000) >>> 24) - 8;
        array[offset + 2] = ((value & 0b0000_0000_1111_0000_0000_0000_0000_0000) >>> 20) - 8;
        array[offset + 3] = ((value & 0b0000_0000_0000_1111_0000_0000_0000_0000) >>> 16) - 8;
        array[offset + 4] = ((value & 0b0000_0000_0000_0000_1111_0000_0000_0000) >>> 12) - 8;
        array[offset + 5] = ((value & 0b0000_0000_0000_0000_0000_1111_0000_0000) >>>  8) - 8;
        array[offset + 6] = ((value & 0b0000_0000_0000_0000_0000_0000_1111_0000) >>>  4) - 8;
        array[offset + 7] =  (value & 0b0000_0000_0000_0000_0000_0000_0000_1111)         - 8;
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

    public static boolean[] bytesToBoolean(byte[] values)
    {
        var result = new boolean[values.length];

        for (int i = 0; i < values.length; i++)
        {
            result[i] = values[i] != -127;
        }

        return result;
    }

    public static byte booleansToByte(boolean[] values)
    {
        int result = 0;
        if (values[0]) result += 128;
        if (values[1]) result += 64;
        if (values[2]) result += 32;
        if (values[3]) result += 16;
        if (values[4]) result += 8;
        if (values[5]) result += 4;
        if (values[6]) result += 2;
        if (values[7]) result += 1;

        return (byte)(result - 128);
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
