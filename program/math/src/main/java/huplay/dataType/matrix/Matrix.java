package huplay.dataType.matrix;

import huplay.dataType.DataType;
import huplay.dataType.vector.Vector;

public interface Matrix
{
    float getValue(int row, int col);

    void setValue(int row, int col, float value);

    Vector getRow(int row);

    void setRow(int row, Vector vector);

    Vector[] getVectorArray();

    int getRowCount();

    int getColCount();

    DataType getInternalFloatType();

    static MatrixType getInternalMatrixType(DataType floatType)
    {
        return switch (floatType)
        {
            case FLOAT_32           -> MatrixType.VECTOR_ARRAY_FLOAT_32;
            case FLOAT_16           -> MatrixType.VECTOR_ARRAY_FLOAT_16;
            case BRAIN_FLOAT_16     -> MatrixType.VECTOR_ARRAY_BRAIN_FLOAT_16;
            default
                    -> throw new RuntimeException("Unsupported data type at emptyVector: " + floatType);
        };
    }

    static Matrix emptyMatrix(DataType floatType, int rows, int cols)
    {
        return emptyMatrix(getInternalMatrixType(floatType), rows, cols);
    }

    static Matrix emptyMatrix(MatrixType matrixType, int rows, int cols)
    {
        return switch (matrixType)
        {
            case VECTOR_ARRAY_FLOAT_32          -> new VectorArrayMatrix(DataType.FLOAT_32, rows, cols);
            case VECTOR_ARRAY_FLOAT_16          -> new VectorArrayMatrix(DataType.FLOAT_16, rows, cols);
            case VECTOR_ARRAY_BRAIN_FLOAT_16    -> new VectorArrayMatrix(DataType.BRAIN_FLOAT_16, rows, cols);
        };
    }
}
