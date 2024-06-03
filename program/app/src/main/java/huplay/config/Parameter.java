package huplay.config;

public class Parameter
{
    private final String id;
    private final ParameterType parameterType;

    public Parameter(String id, ParameterType parameterType)
    {
        this.id = id;
        this.parameterType = parameterType;
    }

    // Getters
    public String getId() {return id;}
    public ParameterType getParameterType() {return parameterType;}
}
