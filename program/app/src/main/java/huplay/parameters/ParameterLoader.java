package huplay.parameters;

import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;

import java.util.HashMap;
import java.util.Map;

/**
 * Loader responsible to load vectors or matrices using a parameter reader
 * The loader and reader are separated to support quantization,
 * where a single weight matrix should be loaded using multiple tensors
 */
public abstract class ParameterLoader
{
    protected final Config config;
    private final Map<String, String> defaultNamingMap = new HashMap<>();

    public ParameterLoader(Config config)
    {
        this.config = config;
    }

    protected void addDefaultNaming(String key, String value)
    {
        defaultNamingMap.put(key, value);
    }

    public abstract Vector loadVector(ParameterReader reader, String parameterId, int size);

    public abstract Matrix loadMatrix(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols);

    public abstract boolean[][] loadBoolArray(ParameterReader reader, String parameterId, int rows, int cols);

    public abstract long calculateByteSize(ParameterReader reader, String parameterId, int size);

    public abstract long calculateByteSize(ParameterReader reader, String parameterId, int rows, int cols);

    protected String getFinalParameterId(String parameterId, String name)
    {
        var naming = config.getQuantizationConfig().getNaming();

        name = naming != null ? naming.get(name) : defaultNamingMap.get(name);

        name = name.replace("{name}", parameterId);

        if (name.contains("{name-1}"))
        {
            if (parameterId.contains("."))
            {
                var lastIndex = parameterId.lastIndexOf(".");
                name = name.replace("{name-1}", parameterId.substring(0, lastIndex));
            }
            else
            {
                name = name.replace("{name-1}", parameterId);
            }
        }

        return name;
    }
}
