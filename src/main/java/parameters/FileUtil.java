package parameters;

import app.IdentifiedException;
import config.RepoConfig;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.round;

public class FileUtil
{
    public static String determineDownloadUrl(RepoConfig repoConfig, String fileName)
    {
        var repoUrl = repoConfig.getRepo();
        var branch = repoConfig.getBranch();

        if (repoUrl.startsWith("https://huggingface.co/"))
        {
            if (branch == null || branch.isEmpty())
            {
                branch = "main";
            }

            branch = branch.replace("/", "%2F");

            repoUrl += "/resolve/" + branch + "/" + fileName + "?download=true";
        }
        else if (repoUrl.startsWith("https://github.com/"))
        {
            if (branch == null || branch.isEmpty())
            {
                branch = "master";
            }

            repoUrl += "/raw/" + branch + "/" + fileName;
        }

        return repoUrl;
    }

    public static boolean checkHeaderFiles(String downloadPath, String headerPath)
    {
        // Checks weather the header folder exists
        File headerFolder = new File(headerPath);
        if (!headerFolder.exists() || !headerFolder.isDirectory())
        {
            return false;
        }

        // Check all header files are created for all safetensors files
        for (var file : new File(downloadPath).listFiles())
        {
            if (file.isFile() && file.getName().endsWith("safetensors"))
            {
                File headerFile = new File(headerPath + "/" + file.getName() + ".header");

                if (!headerFile.exists() && !headerFile.isFile())
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static List<String> checkFiles(RepoConfig tokenizerConfig, String modelPath)
    {
        var missingFiles = new ArrayList<String>();

        for (var fileName : tokenizerConfig.getFiles())
        {
            var path = modelPath + "/" + fileName;

            var file = new File(path);
            if (!file.exists())
            {
                missingFiles.add(fileName);
            }
        }

        return missingFiles;
    }

    public static String readTextFile(String fileName)
    {
        try
        {
            return Files.readString(Paths.get(fileName));
        }
        catch (Exception e)
        {
            throw new IdentifiedException("File read error. (" + fileName + ")", e);
        }
    }

    public static String formatSize(long size)
    {
        long x = 1024;

        if (size < x) return size + " Bytes";
        else if (size < x*x) return round((float)size/x) + " kB";
        else if (size < x*x*x) return round((float)size/x/x) + " MB";
        else if (size < x*x*x*x) return round((float)size/x/x/x) + " GB";
        else return round((float)size/x/x/x/x) + " TB";
    }
}
