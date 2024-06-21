package position.sinusoid;

import config.Config;
import math.dataType.DataType;
import math.dataType.matrix.Matrix;
import math.dataType.vector.Vector;

import static math.BasicMathUtility.exp;
import static math.dataType.matrix.Matrix.emptyMatrix;
import static math.dataType.vector.Vector.emptyVector;

public class SinusoidPositionEmbedding
{
    private Matrix positionMatrix;

    public void init(Config config, int vectorSize)
    {
        var contextSize = config.getContextSize();

        positionMatrix = emptyMatrix(DataType.FLOAT_32, contextSize, vectorSize);

        Vector positions = emptyVector(contextSize);
        for (int i = 0; i < contextSize; i++)
        {
            positions.set(i, i);
        }

        Vector progression = emptyVector(contextSize / 2);
        for (int i = 0; i < contextSize / 2; i++)
        {
            progression.set(i, exp(-i * Math.log(10000) / contextSize));
        }

        for (int pos = 0; pos < contextSize; pos++)
        {
            for (int k = 0; k < vectorSize / 2; k++)
            {
                int i = 2 * k;
                positionMatrix.setValue(pos, i, (float) Math.sin(positions.get(i) * progression.get(k)));
                positionMatrix.setValue(pos, i + 1, (float) Math.sin(positions.get(i + 1) * progression.get(k)));
            }
        }
    }

    public Vector apply(Vector vector, int pos)
    {
        for (int i = 0; i < vector.size(); i++)
        {
            vector.set(i, vector.get(i) * positionMatrix.getValue(pos, i));
        }

        return vector;
    }
}
