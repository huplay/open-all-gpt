package huplay.quantization;

import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.ParameterReader;

public interface Quantizer
{
    Matrix load(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols);

    QuantizedMatrix quantize(ParameterType parameterType, Matrix matrix);
}
