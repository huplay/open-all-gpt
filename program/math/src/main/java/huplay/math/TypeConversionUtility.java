package huplay.math;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

public class TypeConversionUtility
{
    private static final Map<Integer, Integer> maskBases = new HashMap<>(5);

    static
    {
        maskBases.put(1, 0b1);
        maskBases.put(2, 0b11);
        maskBases.put(4, 0b1111);
        maskBases.put(8, 0b1111_1111);
        maskBases.put(16, 0b1111_1111_1111_1111);
        maskBases.put(32, 0b1111_1111_1111_1111_1111_1111_1111_1111);
    }

    /**
     * Unpacks an int (32 bit) value into "n" signed int values of "bits" size,
     * and appends it into an array at the given offset
     * (Of course (n * size) should be 32. No checks here to make it fast.)
     */
    public static void unpackInt(int value, int n, int size, int[] array, int offset)
    {
        var shift = 32 - size;
        for (var i = 0; i < n; i++)
        {
            int mask = maskBases.get(size) << shift;
            array[i + offset] = (byte) ((value & mask) >>> shift) - 8;

            shift -= size;
        }
    }

    /**
     * Unpacks two 4-bit unsigned values of an unsigned byte
     */
    public static void unpack4bitsFromUnsignedByte(byte value, byte[] array, int offset)
    {
        int mask = maskBases.get(4);
        array[offset] = (byte)((value & (mask << 4)) >>> 4);
        array[offset + 1] = (byte)(value & mask);
    }

    public static byte getLower4bitsFromUnsignedByte(byte value)
    {
        return (byte)((value & 0b11110000) >>> 4);
    }

    public static byte getUpper4bitsFromUnsignedByte(byte value)
    {
        return (byte)(value & 0b1111);
    }

    public static int[][] unpackIntMatrixByRow(int[][] matrix, int n, int size)
    {
        int[][] result = new int[matrix.length][matrix[0].length * n];

        int i = 0;
        for (int[] row : matrix)
        {
            var offset = 0;
            for (int value : row)
            {
                unpackInt(value, n, size, result[i], offset);
                offset += n;
            }
            i++;
        }

        return result;
    }

    public static int[][] unpackIntMatrixByCol(int[][] matrix, int n, int size)
    {
        int[][] result = new int[matrix.length * n][matrix[0].length];

        int rowOffset = 0;
        for (int row = 0; row < matrix.length; row++)
        {
            for (var col = 0; col < matrix[0].length; col++)
            {
                int[] array = new int[n];
                unpackInt(matrix[row][col], n, size, array, 0);

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
