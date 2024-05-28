package huplay.parameters;

import huplay.IdentifiedException;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.dataType.DataType;

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
    public Matrix readMatrix(ParameterReader reader, ParameterType parameterType, String id, int rows, int cols)
    {
        DataType FloatType = reader.getDataType(id);

        return switch (FloatType)
        {
            case FLOAT_16           -> reader.readFloat16Matrix(id, rows, cols);
            case BRAIN_FLOAT_16     -> reader.readBrainFloat16Matrix(id, rows, cols);
            case FLOAT_32           -> reader.readFloat32Matrix(id, rows, cols);
            default ->
                    throw new IdentifiedException("Not supported data type: " + FloatType + ", key: " + id);
        };
    }

    private Vector read(ParameterReader reader, String id, int size)
    {
        DataType FloatType = reader.getDataType(id);

        return switch (FloatType)
        {
            case FLOAT_16           -> reader.readFloat16Vector(id, size);
            case BRAIN_FLOAT_16     -> reader.readBrainFloat16Vector(id, size);
            case FLOAT_32           -> reader.readFloat32Vector(id, size);
            default ->
                    throw new IdentifiedException("Not supported data type: " + FloatType + ", key: " + id);
        };
    }
}
