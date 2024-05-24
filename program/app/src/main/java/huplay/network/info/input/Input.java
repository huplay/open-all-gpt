package huplay.network.info.input;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = TokenInput.class, name = "TokenInput"),
        @JsonSubTypes.Type(value = HiddenStateInput.class, name = "HiddenSizeInput")
})
@JsonIgnoreProperties(ignoreUnknown = true)
public interface Input
{
}
