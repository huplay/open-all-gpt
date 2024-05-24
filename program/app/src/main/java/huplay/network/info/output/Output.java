package huplay.network.info.output;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = EmptyOutput.class, name = "EmptyOutput"),
        @JsonSubTypes.Type(value = HiddenStateOutput.class, name = "HiddenSizeOutput"),
        @JsonSubTypes.Type(value = TokenOutput.class, name = "TokenOutput")
})
@JsonIgnoreProperties(ignoreUnknown = true)
public interface Output
{
}
