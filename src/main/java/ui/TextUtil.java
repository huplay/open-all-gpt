package ui;

import java.io.PrintStream;
import java.util.Locale;

public class TextUtil
{
    public static String ansi(String code)
    {
        return "\033[" + code;
    }

    public static String ansiC(char code)
    {
        return "\033[" + Color.getCode(code) + "m";
    }

    public static String ansiBC(char code)
    {
        return "\033[" + Color.getBackgroundCode(code) + "m";
    }

    public static void clearScreen(PrintStream OUT)
    {
        OUT.print(ansi("H") + ansi("2J"));
        OUT.flush();
    }

    public static void clearScreen(PrintStream OUT, char backgroundColor)
    {
        OUT.print(ansiBC(backgroundColor));
        OUT.flush();
        OUT.print(ansi("H") + ansi("2J") + ansiBC(backgroundColor));
        OUT.flush();
    }

    public static void setColor(PrintStream OUT, char color)
    {
        OUT.print(ansiC(color));
    }

    public static void setBackgroundColor(PrintStream OUT, char color)
    {
        OUT.print(ansiBC(color));
    }

    public static String repeat(String text, int n)
    {
        return String.valueOf(text).repeat(Math.max(0, n));
    }

    public static int indexOf(String text, String searched, int nth)
    {
        var index = -1;
        while (nth > 0)
        {
            index = text.indexOf(searched, index + searched.length());
            if (index == -1)
            {
                return -1;
            }
            nth--;
        }

        return index;
    }

    public static boolean equalsIgnoreCase(String a, String b)
    {
        if (a == null || b == null)
        {
            return false;
        }

        return a.toLowerCase(Locale.ROOT).equals(b.toLowerCase(Locale.ROOT));
    }

    public static boolean equalsIgnoreTypo(String a, String b)
    {
        if (a == null || b == null)
        {
            return false;
        }

        a = removeIrrelevantChars(a);
        b = removeIrrelevantChars(b);

        return equalsIgnoreCase(a, b);
    }

    private static String removeIrrelevantChars(String value)
    {
        var irrelevantChars = new String[] {" ", "_", "-", "/"};

        for (var irrelevantChar : irrelevantChars)
        {
            if (value != null && !value.isEmpty())
            {
                value = value.replace(irrelevantChar, "");
            }
        }

        return value;
    }

    public static int readInt(String value, int defaultValue)
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            System.out.println("\nWARNING: The provided value can't be converted to integer (" + value
                    + "). Default value will be used.\n");
        }

        return defaultValue;
    }

    public static String toCenter(String text, int size)
    {
        var length = countCharacters(text);

        var spaces = "";
        if (length < size - 1)
        {
            var n = (size - length) / 2;
            spaces = repeat( " ", n);
        }

        return spaces + text;
    }

    public static boolean hasNonSpace(String line)
    {
        line = line.replace(" ", "");
        line = line.replaceAll("\u001B\\[[;\\d]*m", "");

        return !line.isEmpty();
    }

    public static int countCharacters(String line)
    {
        line = line.replaceAll("\u001B\\[[;\\d]*m", "");

        return line.length();
    }
}
