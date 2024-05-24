package huplay.dataType.matrix;

import huplay.dataType.FloatType;
import huplay.dataType.vector.Vector;

import static huplay.dataType.FloatType.*;

public interface Matrix
{
    Vector[] getVectorArray();

    Vector getVector(int index);

    void setVector(int index, Vector vector);

    float getValue(int row, int col);

    void setValue(int row, int col, float value);

    int getRowCount();

    int getColCount();

    FloatType getInternalFloatType();

    static MatrixType getInternalMatrixType(FloatType floatType)
    {
        return switch (floatType)
        {
            case FLOAT_32           -> MatrixType.VECTOR_ARRAY_FLOAT_32;
            case FLOAT_16           -> MatrixType.VECTOR_ARRAY_FLOAT_16;
            case BRAIN_FLOAT_16     -> MatrixType.VECTOR_ARRAY_BRAIN_FLOAT_16;
        };
    }

    static Matrix emptyMatrix(FloatType floatType, int rows, int cols)
    {
        return emptyMatrix(getInternalMatrixType(floatType), rows, cols);
    }

    static Matrix emptyMatrix(MatrixType matrixType, int rows, int cols)
    {
        return switch (matrixType)
        {
            case VECTOR_ARRAY_FLOAT_32          -> new VectorArrayMatrix(FLOAT_32, rows, cols);
            case VECTOR_ARRAY_FLOAT_16          -> new VectorArrayMatrix(FLOAT_16, rows, cols);
            case VECTOR_ARRAY_BRAIN_FLOAT_16    -> new VectorArrayMatrix(BRAIN_FLOAT_16, rows, cols);
            default -> null;
        };
    }
}
