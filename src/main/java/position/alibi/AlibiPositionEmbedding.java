package position.alibi;


/**
  ALiBi position embedding (Attention with Linear Biases)

  Publication (27 Aug 2021): https://arxiv.org/abs/2108.12409
 */
public class AlibiPositionEmbedding
{
    private float[] positionSlope;

    /**
     * Initializes the position slope: equally incremented values between 1/headcount and 1
     */
    public void init(int headCount)
    {
        positionSlope = new float[headCount];

        float step = 1f / headCount;
        for (int i = 0; i < headCount; i++)
        {
            positionSlope[i] = step * (i + 1);
        }
    }

    /**
     * The position is applied as removing a value of the score.
     * This value is proportional to the distance of the token,
     * so more distant tokens will have a lower score.
     */
    public float apply(float score, int head, int relativePos)
    {
        return score - positionSlope[head] * (relativePos - 1);
    }
}
