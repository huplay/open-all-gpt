package huplay.parameters;

import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;

/**
 * Loader responsible to load vectors or matrices using a parameter reader
 * The loader and reader are separated to support quantization,
 * where a single weight matrix should be loaded using multiple tensors
 */
public interface ParameterLoader
{
    Vector readVector(ParameterReader reader, String file, int size);

    Matrix readMatrix(ParameterReader reader, ParameterType parameterType, String file, int rows, int cols);

    long calculateByteSize(ParameterReader reader, String name, int size);
}
