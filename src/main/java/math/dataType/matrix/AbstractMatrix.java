package math.dataType.matrix;

import static math.MathUtil.MATH;
import math.dataType.vector.Vector;

public abstract class AbstractMatrix implements Matrix
{
    @Override
    public Vector flatten()
    {
        return MATH.flattenMatrix(this);
    }
}
