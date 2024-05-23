package huplay.dataType.matrix;

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

    static Matrix emptyMatrix(int rows, int cols)
    {
        return emptyMatrix(MatrixType.VECTOR_ARRAY_FLOAT_32, rows, cols);
    }

    static Matrix emptyMatrix(MatrixType matrixType, int rows, int cols)
    {
        return switch (matrixType)
        {
            case VECTOR_ARRAY_FLOAT_32          -> new VectorArrayMatrix(FLOAT_32, rows, cols);
            case VECTOR_ARRAY_FLOAT_16          -> new VectorArrayMatrix(FLOAT_16, rows, cols);
            case VECTOR_ARRAY_BRAIN_FLOAT_16    -> new VectorArrayMatrix(BRAIN_FLOAT_16, rows, cols);
            case QLoRA_NORMAL_FLOAT_4           -> null;
            case QLoRA_NORMAL_FLOAT_4_DOUBLE    -> null;
        };
    }

    static Matrix of(Vector[] values)
    {
        return of(MatrixType.VECTOR_ARRAY_FLOAT_32, values);
    }

    static Matrix of(MatrixType matrixType, Vector[] values)
    {
        return switch (matrixType)
        {
            case VECTOR_ARRAY_FLOAT_32          -> new VectorArrayMatrix(FLOAT_32, values);
            case VECTOR_ARRAY_FLOAT_16          -> new VectorArrayMatrix(FLOAT_16, values);
            case VECTOR_ARRAY_BRAIN_FLOAT_16    -> new VectorArrayMatrix(BRAIN_FLOAT_16, values);
            case QLoRA_NORMAL_FLOAT_4           -> null;
            case QLoRA_NORMAL_FLOAT_4_DOUBLE    -> null;
        };
    }
}
