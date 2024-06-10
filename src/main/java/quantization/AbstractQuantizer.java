package quantization;

import config.Config;
import config.ParameterType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;
import parameters.ParameterLoader;
import parameters.ParameterReader;

public abstract class AbstractQuantizer extends ParameterLoader implements Quantizer
{
    public AbstractQuantizer(Config config)
    {
        super(config);
    }

    @Override
    public Matrix loadMatrix(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols)
    {
        return load(reader, parameterType, parameterId, rows, cols);
    }

    @Override
    public Vector loadVector(ParameterReader reader, String parameterId, int size)
    {
        System.out.println("WARNING: Reading quantized vector isn't supported. (Only matrix.)");
        return null;
    }

    @Override
    public boolean[][] loadBoolArray(ParameterReader reader, String parameterId, int rows, int cols)
    {
        System.out.println("WARNING: Reading quantized bool array isn't supported.");
        return null;
    }
}
