package huplay.file;

import huplay.dataType.vector.Vector;

public interface ParameterReader
{
    Vector readFloat32(String id, int size);

    Vector readFloat16(String id, int size);

    Vector readBrainFloat16(String id, int size);

    DataType getDataType(String id);

    long getBits(String id);
}
