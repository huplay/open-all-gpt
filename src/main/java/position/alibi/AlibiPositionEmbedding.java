package position.alibi;

public class AlibiPositionEmbedding
{
    private float[] positionSlope;

    public void init(int headCount)
    {
        positionSlope = new float[headCount];

        float step = 1f / headCount;
        for (int i = 0; i < headCount; i++)
        {
            positionSlope[i] = step * (i + 1);
        }
    }

    public float apply(float score, int head, int relativePos)
    {
        return score - positionSlope[head] * (relativePos - 1);
    }
}
