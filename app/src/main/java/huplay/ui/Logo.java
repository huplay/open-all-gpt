package huplay.ui;

import java.io.PrintStream;

import static huplay.ui.TextUtil.*;

public class Logo
{
    private static final String[] LETTER_ROWS = new String[6];

    static
    {
        // The letters are based on the Standard font of this site: https://patorjk.com/software/taag
        // In this structure the "I" separates the letters, backslash is replaced by "x".
        LETTER_ROWS[0] = "I    I _ I _ _I   _  _   I  _  I _  __I  ___   I _ I  __I__  I      I       I   I       I   I    __I  ___  I _ I ____  I _____ I _  _   I ____  I  __   I _____ I  ___  I  ___  I   I   I  __I       I__  I ___ I   ____  I    _    I ____  I  ____ I ____  I _____ I _____ I  ____ I _   _ I ___ I     _ I _  __I _     I __  __ I _   _ I  ___  I ____  I  ___  I ____  I ____  I _____ I _   _ I__     __I__        __I__  __I__   __I _____I __ I__    I __ I /x I       I _ I       I _     I      I     _ I      I  __ I       I _     I _ I   _ I _    I _ I           I       I       I       I       I      I     I _   I       I       I          I      I       I     I   __I _ I__   I     I";
        LETTER_ROWS[1] = "I    I| |I( | I _| || |_ I | | I(_)/ /I ( _ )  I( )I / /Ix x I__/x__I   _   I   I       I   I   / /I / _ x I/ |I|___ x I|___ / I| || |  I| ___| I / /_  I|___  |I ( _ ) I / _ x I _ I _ I / /I _____ Ix x I|__ xI  / __ x I   / x   I| __ ) I / ___|I|  _ x I| ____|I|  ___|I / ___|I| | | |I|_ _|I    | |I| |/ /I| |    I|  x/  |I| x | |I / _ x I|  _ x I / _ x I|  _ x I/ ___| I|_   _|I| | | |Ix x   / /Ix x      / /Ix x/ /Ix x / /I|__  /I| _|Ix x   I|_ |I|/x|I       I( )I  __ _ I| |__  I  ___ I  __| |I  ___ I / _|I  __ _ I| |__  I(_)I  (_)I| | __I| |I _ __ ___  I _ __  I  ___  I _ __  I  __ _ I _ __ I ___ I| |_ I _   _ I__   __I__      __I__  __I _   _ I ____I  / /I| |Ix x  I     I";
        LETTER_ROWS[2] = "I    I| |I V VI|_  ..  _|I/ __)I  / / I / _ x/xI|/ I| | I | |Ix    /I _| |_ I   I _____ I   I  / / I| | | |I| |I  __) |I  |_ x I| || |_ I|___ x I| '_ x I   / / I / _ x I| (_) |I(_)I(_)I/ / I|_____|I x xI  / /I / / _` |I  / _ x  I|  _ x I| |    I| | | |I|  _|  I| |_   I| |  _ I| |_| |I | | I _  | |I| ' / I| |    I| |x/| |I|  x| |I| | | |I| |_) |I| | | |I| |_) |Ix___ x I  | |  I| | | |I x x / / I x x /x / / I x  / I x V / I  / / I| | I x x  I | |I    I       I x|I / _` |I| '_ x I / __|I / _` |I / _ xI| |_ I / _` |I| '_ x I| |I  | |I| |/ /I| |I| '_ ` _ x I| '_ x I / _ x I| '_ x I / _` |I| '__|I/ __|I| __|I| | | |Ix x / /Ix x /x / /Ix x/ /I| | | |I|_  /I | | I| |I | | I /x/|I";
        LETTER_ROWS[3] = "I    I|_|I    I|_      _|Ix__ xI / /_ I| (_>  <I   I| | I | |I/_  _xI|_   _|I _ I|_____|I _ I / /  I| |_| |I| |I / __/ I ___) |I|__   _|I ___) |I| (_) |I  / /  I| (_) |I x__, |I _ I _ Ix x I|_____|I / /I |_| I| | (_| |I / ___ x I| |_) |I| |___ I| |_| |I| |___ I|  _|  I| |_| |I|  _  |I | | I| |_| |I| . x I| |___ I| |  | |I| |x  |I| |_| |I|  __/ I| |_| |I|  _ < I ___) |I  | |  I| |_| |I  x V /  I  x V  V /  I /  x I  | |  I / /_ I| | I  x x I | |I    I       I   I| (_| |I| |_) |I| (__ I| (_| |I|  __/I|  _|I| (_| |I| | | |I| |I  | |I|   < I| |I| | | | | |I| | | |I| (_) |I| |_) |I| (_| |I| |   Ix__ xI| |_ I| |_| |I x V / I x V  V / I >  < I| |_| |I / /_I< <  I| |I  > >I|/x/ I";
        LETTER_ROWS[4] = "I    I(_)I    I  |_||_|  I(   /I/_/(_)I x___/x/I   I| | I | |I  x/  I  |_|  I( )I       I(_)I/_/   I x___/ I|_|I|_____|I|____/ I   |_|  I|____/ I x___/ I /_/   I x___/ I   /_/ I(_)I( )I x_xI       I/_/ I (_) I x x__,_|I/_/   x_xI|____/ I x____|I|____/ I|_____|I|_|    I x____|I|_| |_|I|___|I x___/ I|_|x_xI|_____|I|_|  |_|I|_| x_|I x___/ I|_|    I x__x_xI|_| x_xI|____/ I  |_|  I x___/ I   x_/   I   x_/x_/   I/_/x_xI  |_|  I/____|I| | I   x_xI | |I    I _____ I   I x____|I|____/ I x___|I x____|I x___|I|_|  I x__, |I|_| |_|I|_|I _/ |I|_|x_xI|_|I|_| |_| |_|I|_| |_|I x___/ I| .__/ I x__, |I|_|   I|___/I x__|I x____|I  x_/  I  x_/x_/  I/_/x_xI x__, |I/___/I | | I| |I | | I     I";
        LETTER_ROWS[5] = "I    I   I    I          I |_| I      I        I   I x_xI/_/ I      I       I|/ I       I   I      I       I   I       I       I        I       I       I       I       I       I   I|/ I    I       I    I     I  x____/ I         I       I       I       I       I       I       I       I     I       I      I       I        I       I       I       I       I       I       I       I       I         I            I      I       I      I|__|I      I|__|I    I|_____|I   I       I       I      I       I      I     I |___/ I       I   I|__/ I      I   I           I       I       I|_|    I    |_|I      I     I     I       I       I          I      I |___/ I     I  x_xI|_|I/_/  I     I";
    }

    public static void logo(PrintStream OUT, String text, String textColors, char colorAfter)
    {
        logo(OUT, text, textColors, colorAfter, false, 0);
    }

    public static void logoCenter(PrintStream OUT, String text, String textColors, char colorAfter, int width)
    {
        logo(OUT, text, textColors, colorAfter, true, width);
    }

    public static void logo(PrintStream OUT, String text, String textColors, char colorAfter,
                            boolean isCenter, int width)
    {
        var logo = drawLetters(text, textColors);

        for (int i = 0; i < logo.length; i++)
        {
            var line = logo[i];
            if (hasNonSpace(line) || (i > 0 && i < logo.length - 1)) // First and last line isn't displayed if empty
            {
                if (isCenter)
                {
                    line = toCenter(line, width);
                }

                OUT.println(line);
            }
        }

        setColor(OUT, colorAfter);
    }

    private static String[] drawLetters(String text, String textColors)
    {
        var characters = text.toCharArray();
        var colours = textColors.toCharArray();

        var result = new String[] {"", "", "", "", "", ""};

        for (int i = 0; i < characters.length; i++)
        {
            var color = 'W';
            if (colours.length > i)
            {
                color = colours[i];
            }

            result = appendLetter(result, getLetter(characters[i]), Color.getCode(color), true);
        }

        return result;
    }

    private static String[] getLetter(char letter)
    {
        var result = new String[6];

        for (var i = 0; i < result.length; i++)
        {
            // We have characters in the range of ASCII 32-126, which will be the index 0-93
            var letterIndex = letter - 32;
            if (letterIndex >= 0 && letterIndex <= 93)
            {
                var startIndex = indexOf(LETTER_ROWS[i], "I", letterIndex + 1);
                var endIndex = indexOf(LETTER_ROWS[i], "I", letterIndex + 2);
                result[i] = LETTER_ROWS[i].substring(startIndex + 1, endIndex).replace('x', '\\');
            }
        }

        return result;
    }

    private static String[] appendLetter(String[] text, String[] letter, int letterColor, boolean isKerning)
    {
        var result = new String[6];

        for (var i = 0; i < result.length; i++)
        {
            var color = "\033[" + letterColor + "m";
            if (isKerning)
            {
                if (!text[i].isEmpty())
                {
                    var last = text[i].charAt(text[i].length() - 1);
                    if (last == ' ')
                    {
                        result[i] = text[i].substring(0, text[i].length() - 1) + color + letter[i];
                    }
                    else
                    {
                        result[i] = text[i] + color + letter[i].substring(1);
                    }
                }
                else
                {
                    result[i] = color + letter[i];
                }
            }
            else
            {
                result[i] = text[i] + color + letter[i];
            }
        }

        return result;
    }
}
