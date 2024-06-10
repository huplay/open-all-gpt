package tokenizer.gpt;

import app.IdentifiedException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class MergesReader
{
    public static Map<Pair, Integer> readMergesFile(File file, boolean isOmitFirstLine)
    {
        var merges = new HashMap<Pair, Integer>(50000);

        try
        {
            var inputStream = new FileInputStream(file);
            var reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            if (isOmitFirstLine) reader.readLine();

            var i = 0;
            while (true)
            {
                var line = reader.readLine();

                if (line == null) break;

                var pairs = line.split(" ");
                merges.put(new Pair(pairs[0], pairs[1]), i);

                i++;
            }

            reader.close();
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot read merges file: " + file.getName(), e);
        }

        return merges;
    }
}
