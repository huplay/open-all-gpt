package huplay.config;

public enum ParameterType
{
    EMBEDDINGS,
    EMBEDDINGS_BIAS,
    HORIZONTAL_WEIGHT,
    VERTICAL_WEIGHT,
    BIAS,
    NORMALIZATION_WEIGHT,
    NORMALIZATION_BIAS;

    /**
     * Marks if a parameter is a "classic" weight parameter.
     * In many cases these are treated differently to other weights. For example:
     * - oriented vertically
     * - quantized
     */
    public boolean isWeight()
    {
        return this.equals(HORIZONTAL_WEIGHT);
    }

    /**
     * Marks if a matrix is stored in vertical orientation (transposed).
     * Horizontal orientation used at the original models, later the vertical became more common
     */
    public boolean isVertical()
    {
        return this.equals(VERTICAL_WEIGHT);
    }

    public boolean isHorizontal()
    {
        return !isVertical();
    }
}
