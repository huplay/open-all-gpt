package math.dataType.matrix;

import math.dataType.DataType;
import math.dataType.vector.Vector;

public interface Matrix
{
    float getValue(int rowId, int colId);

    void setValue(int rowId, int colId, float value);

    /**
     * Returns a row of the matrix (referencing to the original row instance, if possible)
     */
    Vector row(int rowId);

    void setRow(int rowId, Vector vector);

    Vector[] getVectorArray();

    int getRowCount();

    int getColCount();

    DataType getInternalFloatType();

    Vector flatten();

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
