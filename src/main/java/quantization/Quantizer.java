package quantization;

import config.ParameterType;
import math.dataType.matrix.Matrix;
import parameters.ParameterReader;

public interface Quantizer
{
    Matrix load(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols);

    QuantizedMatrix quantize(ParameterType parameterType, Matrix matrix);
}
