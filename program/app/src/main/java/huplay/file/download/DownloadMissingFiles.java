package huplay.file.download;

import huplay.IdentifiedException;
import huplay.config.RepoConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class DownloadMissingFiles
{
    private final DownloadProgressHandler progressHandler;

    public DownloadMissingFiles(DownloadProgressHandler progressHandler)
    {
        this.progressHandler = progressHandler;
    }

    public void download(List<String> missingFiles, RepoConfig modelConfig, String downloadPath)
            throws Exception
    {
        if (!missingFiles.isEmpty())
        {
            if (modelConfig.getRepo() == null || modelConfig.getRepo().isEmpty())
            {
                throw new IdentifiedException("There are missing files: " + missingFiles);
            }
            else
            {
                // Create folder if missing
                var path = Paths.get(downloadPath);
                try
                {
                    Files.createDirectories(path);
                }
                catch (IOException e)
                {
                    throw new IdentifiedException("Cannot create download folder for missing files: " + missingFiles);
                }

                // Download
                for (String missingFile : missingFiles)
                {
                    var downloadTask = new DownloadTask(modelConfig, missingFile, downloadPath);

                    progressHandler.showFile(missingFile, downloadTask.getSize());

                    var thread = new Thread(downloadTask);
                    thread.start();

                    while (downloadTask.isInProgress())
                    {
                        if (downloadTask.getPieces() > 0)
                        {
                            var pieces = downloadTask.getPieces();
                            var position = downloadTask.getPosition();

                            progressHandler.showProgressBar(pieces, position, 50);

                            Thread.sleep(200);
                        }
                    }

                    // Display a completed progress bar
                    progressHandler.showProgressBar(downloadTask.getPieces(), downloadTask.getPieces(), 50);
                }
            }
        }
    }
}
