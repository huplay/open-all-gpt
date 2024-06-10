package tokenizer.gpt;

import java.util.Objects;

public class Pair
{
    private final String left;
    private final String right;

    public Pair(String left, String right)
    {
        this.left = left;
        this.right = right;
    }

    public String getLeft()
    {
        return left;
    }

    public String getRight()
    {
        return right;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var pair = (Pair) o;
        return Objects.equals(left, pair.left) && Objects.equals(right, pair.right);
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(left, right);
    }
}
