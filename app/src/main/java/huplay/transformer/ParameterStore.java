package huplay.transformer;

import huplay.config.Config;
import huplay.file.SafetensorsReader;
import huplay.config.ParameterType;
import huplay.dataType.vector.Vector;

import java.util.HashMap;
import java.util.Map;

public abstract class ParameterStore
{
    public Config config;
    private long parameterSize;
    private long parameterByteSize;

    public SafetensorsReader reader;

    public final Map<ParameterType, Vector> vectorParams = new HashMap<>();
    public final Map<ParameterType, Vector[]> matrixParams = new HashMap<>();

    public void init(Config config)
    {
        this.config = config;
        this.reader = config.getReader();
    }

    protected abstract String formatName(String file);

    protected void loadVector(ParameterType parameterType, String file, int size)
    {
        var name = formatName(file);
        calculateSize(name, size);

        if (!config.isCalculationOnly())
        {
            vectorParams.put(parameterType, reader.readVector(name, size));
        }
    }

    protected void loadMatrix(ParameterType parameterType, String file, int rows, int cols)
    {
        var name = formatName(file);
        calculateSize(name, rows * cols);

        if (!config.isCalculationOnly())
        {
            matrixParams.put(parameterType, reader.readMatrix(name, rows, cols));
        }
    }

    private void calculateSize(String name, int size)
    {
        var dataType = reader.getDataType(name);

        parameterSize += size;
        parameterByteSize += ((long)size) * dataType.getBits() / 8;
    }

    public Vector vector(ParameterType parameterType)
    {
        return vectorParams.get(parameterType);
    }

    public Vector[] matrix(ParameterType parameterType)
    {
        return matrixParams.get(parameterType);
    }

    // Getters
    public long getParameterSize() {return parameterSize;}
    public long getParameterByteSize() {return parameterByteSize;}
}
