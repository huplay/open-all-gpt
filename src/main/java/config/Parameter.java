package config;

public class Parameter
{
    private final ParameterType parameterType;
    private final String id;

    public Parameter(ParameterType parameterType, String id)
    {
        this.parameterType = parameterType;
        this.id = id;
    }

    // Getters
    public ParameterType getParameterType() {return parameterType;}
    public String getId() {return id;}
}
