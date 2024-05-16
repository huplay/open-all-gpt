package huplay.ui;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

public class ConsoleUtil
{
    public static String input(PrintStream OUT, String text)
    {
        try
        {
            OUT.print(text);
            var reader = new BufferedReader(new InputStreamReader(System.in));
            return reader.readLine();
        }
        catch (IOException e)
        {
            OUT.println("ERROR while reading the input");
            return null;
        }
    }

    public static int intInput(PrintStream OUT, String text, String errorText)
    {
        Integer port = null;

        while (port == null)
        {
            try
            {
                var input = input(OUT, text);
                port = Integer.parseInt(input);
            }
            catch (NumberFormatException e)
            {
                OUT.println(errorText);
            }
        }

        return port;
    }

    public static PrintStream getPrintStream()
    {
        try
        {
            return new PrintStream(System.out, true, StandardCharsets.UTF_8);
        }
        catch (Exception e)
        {
            System.out.println("\nError during setting the console to UTF-8:\n" + e.getMessage());
            return System.out;
        }
    }
}
