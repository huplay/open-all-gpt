package huplay.math;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static huplay.math.TypeConversionUtility.*;

public class TypeConversionUtilityTest
{
    @Test
    public void unpackInt_4Bit_test()
    {
        var input1 = 0b0000_0001_0010_0011_0100_0101_0110_0111;
        var input2 = 0b1000_1001_1010_1011_1100_1101_1110_1111;

        var array = new int[16];

        unpack4bitsFromInt32(input1, array, 0);
        unpack4bitsFromInt32(input2, array, 8);

        assertArrayEquals(new int[]{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, array);
    }

    @Test
    public void unpackIntMatrixByRow_test()
    {
        var input00 = 0b0000_0001_0010_0011_0100_0101_0110_0111; var input01 = 0b1000_1001_1010_1011_1100_1101_1110_1111;
        var input10 = 0b0111_0110_0101_0100_0011_0010_0001_0000; var input11 = 0b1111_1110_1101_1100_1011_1010_1001_1000;

        var matrix = new int[2][2];
        matrix[0][0] = input00; matrix[0][1] = input01;
        matrix[1][0] = input10; matrix[1][1] = input11;

        int[][] result = unpack4bitsFromIntMatrixByRow(matrix);

        assertArrayEquals(new int[][]{
                {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
                {-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0}}, result);
    }

    @Test
    public void unpackIntMatrixByCol_test()
    {
        var input00 = 0b0000_0001_0010_0011_0100_0101_0110_0111; var input01 = 0b1000_1001_1010_1011_1100_1101_1110_1111;
        var input10 = 0b0111_0110_0101_0100_0011_0010_0001_0000; var input11 = 0b1111_1110_1101_1100_1011_1010_1001_1000;

        var matrix = new int[2][2];
        matrix[0][0] = input00; matrix[0][1] = input01;
        matrix[1][0] = input10; matrix[1][1] = input11;

        int[][] result = unpackIntMatrixByCol(matrix, 8, 4);

        assertArrayEquals(new int[][]{
                {-8, 0},
                {-7, 1},
                {-6, 2},
                {-5, 3},
                {-4, 4},
                {-3, 5},
                {-2, 6},
                {-1, 7},
                {-1, 7},
                {-2, 6},
                {-3, 5},
                {-4, 4},
                {-5, 3},
                {-6, 2},
                {-7, 1},
                {-8, 0}
        }, result);
    }
}
