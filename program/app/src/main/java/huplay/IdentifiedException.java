package huplay;

public class IdentifiedException extends RuntimeException
{
    public IdentifiedException(String message, Exception e)
    {
        super(message + " Message: " + e.getMessage(), e);
    }

    public IdentifiedException(String message)
    {
        super(message);
    }
}
