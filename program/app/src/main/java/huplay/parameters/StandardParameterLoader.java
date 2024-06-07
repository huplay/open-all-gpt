package huplay.parameters;

import huplay.IdentifiedException;
import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.dataType.DataType;

public class StandardParameterLoader extends ParameterLoader
{
    public StandardParameterLoader(Config config)
    {
        super(config);
    }

    @Override
    public Vector loadVector(ParameterReader reader, String parameterId, int size)
    {
        return read(reader, parameterId, size);
    }

    @Override
    public Matrix loadMatrix(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols)
    {
        DataType floatType = reader.getDataType(parameterId);
        return readMatrix(reader, floatType, parameterId, rows, cols);
    }

    @Override
    public boolean[][] loadBoolArray(ParameterReader reader, String parameterId, int rows, int cols)
    {
        return reader.readBooleanArray2D(parameterId, rows, cols);
    }

    private Matrix readMatrix(ParameterReader reader, DataType dataType, String parameterId, int rows, int cols)
    {
        return switch (dataType)
        {
            case FLOAT_16 -> reader.readFloat16Matrix(parameterId, rows, cols);
            case BRAIN_FLOAT_16 -> reader.readBrainFloat16Matrix(parameterId, rows, cols);
            case FLOAT_32 -> reader.readFloat32Matrix(parameterId, rows, cols);
            default ->
                    throw new IdentifiedException("Not supported data type: " + dataType + ", key: " + parameterId);
        };
    }

    private Vector read(ParameterReader reader, String parameterId, int size)
    {
        DataType FloatType = reader.getDataType(parameterId);

        return switch (FloatType)
        {
            case FLOAT_16           -> reader.readFloat16Vector(parameterId, size);
            case BRAIN_FLOAT_16     -> reader.readBrainFloat16Vector(parameterId, size);
            case FLOAT_32           -> reader.readFloat32Vector(parameterId, size);
            default ->
                    throw new IdentifiedException("Not supported data type: " + FloatType + ", key: " + parameterId);
        };
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int size)
    {
        return ((long)size) * reader.getBits(parameterId) / 8;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int rows, int cols)
    {
        return rows * (cols * reader.getBits(parameterId) / 8);
    }
}
