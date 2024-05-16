package huplay;

public class IdentifiedException extends RuntimeException
{
    public IdentifiedException(String message, Exception e)
    {
        super(message + " Message: " + e.getStackTrace(), e);
    }

    public IdentifiedException(String message)
    {
        super(message);
    }
}
