package huplay.parameters;

import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;
import huplay.file.ParameterReader;

public interface ParameterLoader
{
    Vector readVector(ParameterReader reader, String file, int size);

    Matrix readMatrix(ParameterReader reader, String file, int rows, int cols);

    long calculateByteSize(ParameterReader reader, String name, int size);
}
