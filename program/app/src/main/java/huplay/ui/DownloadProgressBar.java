package huplay.ui;

import huplay.parameters.download.DownloadProgressHandler;
import huplay.parameters.FileUtil;

import java.io.PrintStream;

import static huplay.ui.Color.*;

public class DownloadProgressBar implements DownloadProgressHandler
{
    // There's no full block unicode character (0x2588 only almost full),
    // so I created it, changing the background colour of a space character
    private static final String FULL = BLUE.getAnsiBC() + " " + WHITE.getAnsiBC();

    // Blocks with growing size (to make the progress more fine-grained)
    private static final char[] BLOCKS = new char[] {' ', 0x258F, 0x258E, 0x258D, 0x258B, 0x258A, 0x2589};

    private final PrintStream OUT;

    public DownloadProgressBar(PrintStream OUT)
    {
        this.OUT = OUT;
    }

    @Override
    public void showFile(String fileName, long size)
    {
        OUT.println("File: " + fileName + " (size: " + FileUtil.formatSize(size) + ")");
    }

    @Override
    public void showProgressBar(long total, long actual, int length)
    {
        if (total == actual)
        {
            // Display a completed progress bar
            var progressBar = FULL.repeat(Math.max(0, length));

            OUT.print("Download: " + BLUE.getAnsiC() + progressBar + RESET + " DONE   \n\n"); // Jump to next line
        }
        else
        {
            // Calculate the actual position within the progress bar and the percentage
            // Make it 7 times bigger to be able to show the progress within a character
            var position = Math.floorDiv(length * actual * 7, total);
            var percentage = String.format("%.2f", (float) 100 * actual / total);

            // Calculate the number of full blocks and the size of the last (progressing) character
            var intPos = Math.floorDiv(position, 7);
            var remainderPos = position % 7;

            var progressBar = new StringBuilder();
            for (var i = 0; i < length; i++)
            {
                if (intPos > i)
                {
                    progressBar.append(FULL); // Display the full characters
                }
                else if (intPos == i)
                {
                    progressBar.append(BLOCKS[(int) remainderPos]); // Display the actually progressing character
                }
                else
                {
                    progressBar.append(" "); // Display the empty characters
                }
            }

            // Use only \r to remain in the same line and redraw it next time
            OUT.print("Download: " + BLUE.getAnsiC() + progressBar + RESET + " " + percentage + "%\r");
        }
    }
}
