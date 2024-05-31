package huplay.parameters;

import huplay.config.Config;
import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.safetensors.SafetensorsReader;
import huplay.config.ParameterType;
import huplay.dataType.vector.Vector;
import huplay.quantization.QuantizationType;

import java.util.HashMap;
import java.util.Map;

public abstract class ParameterStore
{
    public Config config;
    private long parameterSize;
    private long parameterByteSize;
    private DataType internalFloatType;

    public SafetensorsReader reader;

    public final Map<ParameterType, Vector> vectorParams = new HashMap<>();
    public final Map<ParameterType, Matrix> matrixParams = new HashMap<>();

    public void init(Config config)
    {
        this.config = config;
        this.reader = config.getReader();
        this.internalFloatType = config.getInternalFloatType();
    }

    protected Vector emptyVector(int size)
    {
        return Vector.emptyVector(internalFloatType, size);
    }

    protected Matrix emptyMatrix(int rows, int cols)
    {
        return Matrix.emptyMatrix(internalFloatType, rows, cols);
    }

    protected abstract String formatName(String file);

    protected void loadVector(ParameterType parameterType, String id, int size)
    {
        var parameterLoader = getParameterLoader(parameterType, id);

        var name = formatName(id);

        parameterSize += size;
        parameterByteSize += parameterLoader.calculateByteSize(reader, name, size);

        if (!config.isCalculationOnly())
        {
            vectorParams.put(parameterType, parameterLoader.readVector(reader, name, size));
        }
    }

    protected void loadMatrix(ParameterType parameterType, String id, int rows, int cols)
    {
        var parameterLoader = getParameterLoader(parameterType, id);

        var name = formatName(id);

        parameterSize += (long) rows * cols;
        parameterByteSize += parameterLoader.calculateByteSize(reader, name, rows * cols);

        if (!config.isCalculationOnly())
        {
            var parameters = parameterLoader.readMatrix(reader, parameterType, name, rows, cols);
            matrixParams.put(parameterType, parameters);
        }
    }

    private ParameterLoader getParameterLoader(ParameterType parameterType, String id)
    {
        var quantizationConfig = config.getQuantizationConfig();
        if (quantizationConfig != null && quantizationConfig.isQuantized(parameterType, id))
        {
            return QuantizationType.getParameterLoader(config);
        }
        else
        {
            return new StandardParameterLoader(config);
        }
    }

    public Vector vector(ParameterType parameterType)
    {
        return vectorParams.get(parameterType);
    }

    public Matrix matrix(ParameterType parameterType)
    {
        return matrixParams.get(parameterType);
    }

    // Getters
    public long getParameterSize() {return parameterSize;}
    public long getParameterByteSize() {return parameterByteSize;}
}
