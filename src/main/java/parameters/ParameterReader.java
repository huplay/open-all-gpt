package parameters;

import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import parameters.safetensors.SafetensorsHeader;

import java.util.Map;

/**
 * Parameter reader which is responsible to read a specific data as a vector from the parameter files
 */
public interface ParameterReader
{
    Map<String, SafetensorsHeader> getParameterHeaders();

    String readString(String parameterId);

    int[] readIntArray(String parameterId, int size);
    int[][] readIntArray2D(String parameterId, int rows, int cols);

    byte[] readByteArray(String parameterId, int size);
    byte[][] readByteArray2D(String parameterId, int rows, int cols);

    float[] readFloatArray(String parameterId, int size);
    float[][] readFloatArray2D(String parameterId, int rows, int cols);

    short[] readShortArray(String parameterId, int size);
    short[][] readShortArray2D(String parameterId, int rows, int cols);

    boolean[] readBooleanArray(String parameterId, int size);
    boolean[][] readBooleanArray2D(String parameterId, int rows, int cols);

    Vector readFloat32Vector(String parameterId, int size);
    Matrix readFloat32Matrix(String parameterId, int rows, int cols);

    Vector readFloat16Vector(String parameterId, int size);
    Matrix readFloat16Matrix(String parameterId, int rows, int cols);

    Vector readBrainFloat16Vector(String parameterId, int size);
    Matrix readBrainFloat16Matrix(String parameterId, int rows, int cols);

    DataType getDataType(String parameterId);

    long getBits(String parameterId);
}
