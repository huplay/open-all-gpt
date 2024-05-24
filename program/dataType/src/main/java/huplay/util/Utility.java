package huplay.util;

import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.Vector;

import java.util.List;

public interface Utility
{
    String getUtilName();

    /**
     * Vector to vector addition
     */
    Vector addVectors(Vector vector1, Vector vector2);

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    float dotProduct(Vector vector1, Vector vector2);

    /**
     * Multiply vector by a scalar
     */
    Vector mulVectorByScalar(Vector vector, float scalar);

    /**
     * Multiply vector by matrix
     */
    Vector mulVectorByMatrix(Vector vector, Matrix matrix);

    /**
     * Multiply vector by transposed matrix
     */
    Vector mulVectorByTransposedMatrix(Vector vector, Matrix matrix);

    /**
     * Split a vector to a matrix
     */
    Matrix splitVector(Vector vector, int rows);

    /**
     * Merge the rows of a matrix to a single vector
     */
    Vector flattenMatrix(Matrix matrix);

    /**
     * Finds the maximum value in the vector
     */
    float max(Vector vector);

    /**
     * Finds the maximum value in a list of IndexedValue
     */
    float max(List<IndexedValue> vector);

    /**
     * Calculate average (mean) value
     */
    float average(Vector vector);

    /**
     * Calculate the average difference
     */
    float averageDiff(Vector values, float average, float epsilon);

    /**
     * Standard normalization - (value - avg) * sqrt( (value - avg)^2 + epsilon)
     */
    Vector normalize(Vector vector, float epsilon);
}
