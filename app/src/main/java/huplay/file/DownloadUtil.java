package huplay.file;

import huplay.config.RepoConfig;

import java.io.File;

import static java.lang.Math.round;

public class DownloadUtil
{
    public static String determineDownloadUrl(RepoConfig repoConfig, String fileName)
    {
        var repoUrl = repoConfig.getRepo();

        if (repoUrl.startsWith("https://huggingface.co/"))
        {
            var branch = repoConfig.getBranch();

            if (branch == null || branch.isEmpty())
            {
                branch = "main";
            }

            repoUrl += "/resolve/" + branch + "/" + fileName + "?download=true";
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
