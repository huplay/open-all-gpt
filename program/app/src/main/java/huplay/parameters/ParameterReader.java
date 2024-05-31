package huplay.parameters;

import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.parameters.safetensors.SafetensorsHeader;

import java.util.Map;

/**
 * Parameter reader which is responsible to read a specific data as a vector from the parameter files
 */
public interface ParameterReader
{
    Map<String, SafetensorsHeader> getParameterHeaders();

    String readString(String id);

    int[] readIntArray(String id, int size);
    int[][] readIntArray2D(String id, int rows, int cols);

    byte[] readByteArray(String id, int size);
    byte[][] readByteArray2D(String id, int rows, int cols);

    float[] readFloatArray(String id, int size);
    float[][] readFloatArray2D(String id, int rows, int cols);

    short[] readShortArray(String id, int size);
    short[][] readShortArray2D(String id, int rows, int cols);

    Vector readFloat32Vector(String id, int size);
    Matrix readFloat32Matrix(String id, int rows, int cols);

    Vector readFloat16Vector(String id, int size);
    Matrix readFloat16Matrix(String id, int rows, int cols);

    Vector readBrainFloat16Vector(String id, int size);
    Matrix readBrainFloat16Matrix(String id, int rows, int cols);

    DataType getDataType(String id);

    long getBits(String id);
}
