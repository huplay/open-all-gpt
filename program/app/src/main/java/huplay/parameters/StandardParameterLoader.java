package huplay.parameters;

import huplay.IdentifiedException;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.file.DataType;
import huplay.file.ParameterReader;

import static huplay.MathUtilProvider.MATH;

public class StandardParameterLoader implements ParameterLoader
{
    @Override
    public long calculateByteSize(ParameterReader reader, String id, int size)
    {
        return ((long)size) * reader.getBits(id) / 8;
    }

    @Override
    public Vector readVector(ParameterReader reader, String file, int size)
    {
        return read(reader, file, size);
    }

    @Override
    public Matrix readMatrix(ParameterReader reader, String file, int rows, int cols)
    {
        var vector = read(reader, file, rows * cols);
        return vector == null ? null : MATH.splitVector(vector, rows);
    }

    private Vector read(ParameterReader reader, String id, int size)
    {
        DataType datatType = reader.getDataType(id);

        return switch (datatType)
        {
            case FLOAT_16    -> reader.readFloat16(id, size);
            case BRAIN_FLOAT_16   -> reader.readBrainFloat16(id, size);
            case FLOAT_32    -> reader.readFloat32(id, size);
            default ->
                    throw new IdentifiedException("Not supported data type: " + datatType + ", key: " + id);
        };
    }
}
