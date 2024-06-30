package position.sinusoid;

import config.Config;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import static math.BasicMathUtility.*;
import static math.dataType.matrix.Matrix.emptyMatrix;
import static math.dataType.vector.Vector.emptyVector;

public class SinusoidPositionEmbedding
{
    private Matrix positionMatrix;

    public void initInterleaved(Config config, int vectorSize)
    {
        init(config, vectorSize, true);
    }

    public void initSliced(Config config, int vectorSize)
    {
        init(config, vectorSize, false);
    }

    private void init(Config config, int vectorSize, boolean isInterleaved)
    {
        var positionCount = config.getContextSize() + 2;

        positionMatrix = emptyMatrix(DataType.FLOAT_32, positionCount, vectorSize);

        // Range between 0 and (contextSize + 2)
        Vector positions = emptyVector(positionCount);
        for (int i = 0; i < positionCount; i++)
        {
            positions.set(i, i);
        }

        // Exponential progression for a range of vectorSize / 2
        int halfSize = vectorSize / 2;
        double emb = Math.log(10000) / (halfSize - 1);

        Vector progression = emptyVector(halfSize);
        for (int i = 0; i < halfSize; i++)
        {
            progression.set(i, exp(i * -emb));
        }

        // Make a matrix of the two ranges
        for (int pos = 0; pos < positionCount; pos++)
        {
            for (int i = 0; i < halfSize; i++)
            {
                var value = positions.get(pos) * progression.get(i);

                if (isInterleaved)
                {
                    positionMatrix.setValue(pos, 2 * i,     sin(value));
                    positionMatrix.setValue(pos, 2 * i + 1, cos(value));
                }
                else
                {
                    positionMatrix.setValue(pos, i,            sin(value));
                    positionMatrix.setValue(pos, i + halfSize, cos(value));
                }
            }
        }
    }

    public Vector apply(Vector vector, int pos)
    {
        Vector result = emptyVector(vector.getFloatType(), vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            result.set(i, vector.get(i) + positionMatrix.getValue(pos, i));
        }

        return result;
    }
}
