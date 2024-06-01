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
        DataType floatType = reader.getDataType(id);

        var matrix = readMatrix(reader, floatType, id, rows, cols);

        /*if (parameterType.isWeight())
        {
            // TODO: This is just a temporary experiment:
            // Quantize the model on-the fly
            matrix = quantize(matrix);
        }*/

        return matrix;
    }

    private Matrix readMatrix(ParameterReader reader, DataType floatType, String id, int rows, int cols)
    {
        return switch (floatType)
        {
            case FLOAT_16 -> reader.readFloat16Matrix(id, rows, cols);
            case BRAIN_FLOAT_16 -> reader.readBrainFloat16Matrix(id, rows, cols);
            case FLOAT_32 -> reader.readFloat32Matrix(id, rows, cols);
            default ->
                    throw new IdentifiedException("Not supported data type: " + floatType + ", key: " + id);
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
