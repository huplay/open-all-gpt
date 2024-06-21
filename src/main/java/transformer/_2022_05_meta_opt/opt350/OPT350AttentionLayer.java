package transformer._2022_05_meta_opt.opt350;

import math.dataType.vector.Vector;
import transformer._2022_05_meta_opt.OPTAttentionLayer;

import static math.MathUtil.MATH;

/**
 * Meta (Facebook) OPT-350 decoder (attention block) implementation
 * Only the process method is re-implemented, the rest is inherited from the standard OPT implementation
 *
 * @author Hunor Szegi
 */
public class OPT350AttentionLayer extends OPTAttentionLayer
{
    public Vector process(Vector inputHiddenState, boolean isInputOnly)
    {
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        // Not necessary to do the remaining if processing an input token (except the last) and it is the last decoder
        if ( !(isInputOnly && lastDecoder) )
        {
            // Residual connection
            hiddenState = hiddenState.add(inputHiddenState);

            // Normalization
            hiddenState = MATH.layerNorm(hiddenState, vector(normWeight), vector(normBias), epsilon);
        }

        return hiddenState;
    }
}
