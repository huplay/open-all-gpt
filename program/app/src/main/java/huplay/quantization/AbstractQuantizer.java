package huplay.quantization;

import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.parameters.ParameterLoader;
import huplay.parameters.ParameterReader;

public abstract class AbstractQuantizer extends ParameterLoader implements huplay.quantization.Quantizer
{
    public AbstractQuantizer(Config config)
    {
        super(config);
    }

    @Override
    public Matrix loadMatrix(ParameterReader reader, ParameterType parameterType, String id, int rows, int cols)
    {
        return load(reader, parameterType, id, rows, cols);
    }

    @Override
    public Vector loadVector(ParameterReader reader, String file, int size)
    {
        System.out.println("WARNING: Reading quantized vector isn't supported. (Only matrix.)");
        return null;
    }
}
