package huplay.dataType.matrix;

import huplay.MathProvider;
import huplay.dataType.vector.Vector;

public abstract class AbstractMatrix implements Matrix
{
    @Override
    public Vector flatten()
    {
        return MathProvider.getMathUtility().flattenMatrix(this);
    }
}
