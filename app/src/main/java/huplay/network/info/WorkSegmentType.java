package huplay.network.info;

import com.fasterxml.jackson.annotation.JsonIgnore;
import huplay.network.info.input.HiddenStateInput;
import huplay.network.info.input.TokenInput;
import huplay.network.info.output.HiddenStateOutput;
import huplay.network.info.output.TokenOutput;

public enum WorkSegmentType
{
    // The whole Transformer processing should be done by a single worker
    FULL(TokenInput.class, TokenOutput.class),

    // Only the Transformer preProcess should be done
    HEAD_ONLY(TokenInput.class, HiddenStateOutput.class),

    // The Transformer preProcess plus one or more decoder blocks should be processed
    HEAD_AND_LAYERS(TokenInput.class, HiddenStateOutput.class),

    // Only decoder blocks should be processed
    LAYERS_ONLY(HiddenStateInput.class, HiddenStateOutput.class),

    // Some layers plus the post process should be executed
    LAYERS_AND_TAIL(HiddenStateInput.class, TokenOutput.class),

    // Only the Transformer postProcess should be done
    TAIL_ONLY(HiddenStateInput.class, TokenOutput.class);

    private final Class<?> inputClass;
    private final Class<?> outputClass;

    WorkSegmentType(Class<?> inputClass, Class<?> outputClass)
    {
        this.inputClass = inputClass;
        this.outputClass = outputClass;
    }

    // Getters
    @JsonIgnore public Class<?> getInputClass() {return inputClass;}
    @JsonIgnore public Class<?> getOutputClass() {return outputClass;}

    @JsonIgnore
    public boolean hasHead()
    {
        return this.equals(FULL) || this.equals(HEAD_ONLY) || this.equals(HEAD_AND_LAYERS);
    }

    @JsonIgnore
    public boolean hasLayer()
    {
        return this.equals(FULL) || this.equals(HEAD_AND_LAYERS) || this.equals(LAYERS_ONLY) || this.equals(LAYERS_AND_TAIL);
    }

    @JsonIgnore
    public boolean hasTail()
    {
        return this.equals(FULL) || this.equals(LAYERS_AND_TAIL) || this.equals(TAIL_ONLY);
    }
}
