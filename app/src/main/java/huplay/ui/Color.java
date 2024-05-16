package huplay.ui;

import java.util.Arrays;

public enum Color
{
    BLACK('-', 30, 40),
    RED('R', 31, 41),
    GREEN('G', 32, 42),
    YELLOW('Y', 33, 43),
    BLUE('B', 34, 44),
    MAGENTA('M', 35, 45),
    CYAN('C', 36, 46),
    WHITE('W', 37, 47),
    GREY('X', 90, 100),
    BRIGHT_RED('r', 91, 101),
    BRIGHT_GREEN('g', 92, 102),
    BRIGHT_YELLOW('y', 93, 103),
    BRIGHT_BLUE('b', 94, 104),
    BRIGHT_MAGENTA('m', 95, 105),
    BRIGHT_CYAN('c', 96, 106),
    BRIGHT_WHITE('w', 97, 107);

    public static final String RESET = "\033[0m"; // Changes the colour and background colour to default

    private final char abbreviation;
    private final int code;
    private final int backgroundCode;

    Color(char abbreviation, int code, int backgroundCode)
    {
        this.abbreviation = abbreviation;
        this.code = code;
        this.backgroundCode = backgroundCode;
    }

    // Getters
    public char getAbbreviation() {return abbreviation;}
    public int getCode() {return code;}
    public int getBackgroundCode() {return backgroundCode;}

    public String getAnsiC()
    {
        return "\033[" + code + "m";
    }

    public String getAnsiBC()
    {
        return "\033[" + backgroundCode + "m";
    }

    public static int getCode(char abbreviation)
    {
        return Arrays.stream(Color.values())
                .filter(c -> c.abbreviation == abbreviation)
                .map(c -> c.code)
                .findFirst()
                .orElse(WHITE.getCode());
    }

    public static int getBackgroundCode(char abbreviation)
    {
        return Arrays.stream(Color.values())
                .filter(c -> c.abbreviation == abbreviation)
                .map(c -> c.backgroundCode)
                .findFirst()
                .orElse(WHITE.getBackgroundCode());
    }
}
