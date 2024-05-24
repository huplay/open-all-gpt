package huplay.parameters.quantization.gptq;

import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.file.ParameterReader;
import huplay.parameters.ParameterLoader;

public class GPTQParameterLoader implements ParameterLoader
{
    @Override
    public Vector readVector(ParameterReader reader, String file, int size)
    {
        return null;
    }

    @Override
    public Matrix readMatrix(ParameterReader reader, String file, int rows, int cols)
    {
        return null;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String name, int size)
    {
        return 0;
    }
}
