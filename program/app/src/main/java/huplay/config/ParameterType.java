package huplay.config;

import static huplay.config.ParameterType.Constants.*;

public enum ParameterType
{
    // Input
    TOKEN_EMBEDDINGS,
    TOKEN_EMBEDDING_BIAS,
    POSITION_EMBEDDINGS,

    INPUT_NORM_WEIGHT,
    INPUT_NORM_BIAS,

    // Attention block:
    ATT_COMBINED_WEIGHT             (WEIGHT),
    ATT_COMBINED_VERTICAL_WEIGHT    (WEIGHT, true),
    ATT_COMBINED_BIAS,

    ATT_QUERY_WEIGHT                (WEIGHT),
    ATT_QUERY_BIAS,

    ATT_KEY_WEIGHT                  (WEIGHT),
    ATT_KEY_BIAS,

    ATT_VALUE_WEIGHT                (WEIGHT),
    ATT_VALUE_BIAS,

    ATT_PROJ_WEIGHT                 (WEIGHT),
    ATT_VERTICAL_PROJ_WEIGHT        (WEIGHT, true),
    ATT_PROJ_BIAS,

    ATT_NORM_WEIGHT,
    ATT_NORM_BIAS,

    ROTARY_EMBEDDING,

    // Neural net block:
    MLP_1_WEIGHT                    (WEIGHT),
    MLP_1_VERTICAL_WEIGHT           (WEIGHT, true),
    MLP_1_BIAS,

    MLP_2_WEIGHT                    (WEIGHT),
    MLP_2_VERTICAL_WEIGHT           (WEIGHT, true),
    MLP_2_BIAS,

    MLP_3_WEIGHT                    (WEIGHT),
    MLP_3_BIAS,

    MLP_NORM_WEIGHT,
    MLP_NORM_BIAS,

    // Output:
    OUTPUT_NORM_WEIGHT,
    OUTPUT_NORM_BIAS,

    // Test:
    TEST;

    /**
     * Marks if a parameter is a "classic" weight parameter. These parameters are expected to be quantized
     */
    private final boolean isWeight;

    /**
     * Marks if a matrix is stored in vertical orientation (transposed).
     * Horizontal orientation used at the original models, later the vertical became more common
     */
    private final boolean isVertical;

    ParameterType()
    {
        this(null, false);
    }

    ParameterType(Weight isWeight)
    {
        this(isWeight, false);
    }

    ParameterType(boolean isVertical)
    {
        this(null, isVertical);
    }

    ParameterType(Weight isWeight, boolean isVertical)
    {
        this.isWeight = isWeight != null;
        this.isVertical = isVertical;
    }

    // Getters
    public boolean isWeight() {return isWeight;}
    public boolean isVertical() {return isVertical;}

    public boolean isHorizontal()
    {
        return !isVertical;
    }

    // Trick to differentiate between the two boolean constructor parameters, and allow both optional:
    private static class Weight {}
    static class Constants
    {
        static final Weight WEIGHT = new Weight();
    }
}
