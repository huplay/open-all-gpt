package huplay.parameters;

import huplay.config.Config;
import huplay.dataType.DataType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.safetensors.SafetensorsReader;
import huplay.config.ParameterType;
import huplay.dataType.vector.Vector;
import huplay.quantization.QuantizationType;
import huplay.quantization.QuantizedMatrix;

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

    /**
     * Loads a vector parameter (Currently quantization isn't supported for vectors, only for matrices)
     */
    protected void loadVector(ParameterType parameterType, String id, int size)
    {
        // Get the parameter loader (standard or a quantizer, but non-of the quantizers supports vectors)
        var parameterLoader = getParameterLoader(parameterType, id);

        // Resolve the final name of the parameter
        var name = formatName(id);

        // Calculate size
        parameterSize += size;
        parameterByteSize += parameterLoader.calculateByteSize(reader, name, size);

        if (!config.isCalculationOnly())
        {
            // Load and store the vector
            vectorParams.put(parameterType, parameterLoader.loadVector(reader, name, size));
        }
    }

    /**
     * Loads a matrix parameter (standard or quantized)
     * Optionally it can de-quantize a quantized or quantize a non-quantized parameter
     */
    protected void loadMatrix(ParameterType parameterType, String id, int rows, int cols)
    {
        // Get the parameter loader (standard or a quantizer)
        var parameterLoader = getParameterLoader(parameterType, id);

        // Resolve the final name of the parameter
        var name = formatName(id);

        // Calculate size
        parameterSize += (long) rows * cols;
        parameterByteSize += parameterLoader.calculateByteSize(reader, name, rows * cols);

        if (!config.isCalculationOnly())
        {
            // Load the matrix parameter (it can result a standard VectorArrayMatrix, or a quantized matrix as well
            var matrix = parameterLoader.loadMatrix(reader, parameterType, name, rows, cols);

            if (matrix instanceof QuantizedMatrix quantizedMatrix)
            {
                // This is a quantized matrix, ...
                if (config.getQuantizationConfig() != null && config.getQuantizationConfig().getDeQuantizeOnLoad())
                {
                    // ... but we can de-quantize it, if requested
                    matrix = quantizedMatrix.toDeQuantized();
                }
            }
            else
            {
                // This is a non-quantized matrix, ...
                if (parameterType.isWeight() && config.getQuantizeConfig() != null)
                {
                    // ... but we can quantize it, if requested
                    var quantizationType = config.getQuantizeConfig().getQuantizationType();
                    System.out.println("INFO: This parameter is quantized at loading: "  + id + ", using: " + quantizationType);
                    matrix = QuantizationType.getQuantizer(config, quantizationType).quantize(parameterType, matrix);
                }
            }

            // Store the matrix
            matrixParams.put(parameterType, matrix);
        }
    }

    private ParameterLoader getParameterLoader(ParameterType parameterType, String id)
    {
        var quantizationConfig = config.getQuantizationConfig();
        if (quantizationConfig == null || !quantizationConfig.isQuantized(parameterType, id))
        {
            // Get the standard parameter loader to load non-quantized parameters
            return new StandardParameterLoader(config);
        }
        else
        {
            // Get the specific quantizer as parameter loader to load a quantized model
            return QuantizationType.getQuantizer(config, quantizationConfig.getQuantizationType());
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
