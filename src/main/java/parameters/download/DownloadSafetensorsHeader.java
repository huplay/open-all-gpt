package parameters.download;

import app.IdentifiedException;
import config.RepoConfig;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import static parameters.FileUtil.determineDownloadUrl;

public class DownloadSafetensorsHeader
{
    public static void downloadHeader(RepoConfig repoConfig, String downloadPath, String fileName)
    {
        try
        {
            // Create header folder if missing
            var path = Paths.get(downloadPath);
            Files.createDirectories(path);

            // Determine the download url
            var url = new URL(determineDownloadUrl(repoConfig, fileName)); // TODO: Deprecated
            System.out.println("URL: " + url);

            // Download the header
            downloadHeader(url, downloadPath + "/" + fileName + ".header");
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot create download folder: " + downloadPath);
        }
    }

    private static void downloadHeader(URL url, String fileName)
    {
        try
        {
            // Create the download channel
            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());

            // Read the first 8 bytes into a byteBuffer (it is the size of the header)
            var byteBuffer = ByteBuffer.allocateDirect(8);
            urlChannel.read(byteBuffer);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
            long headerSize = byteBuffer.getLong(0);

            // Download the next <headerSize> bytes as a text file
            FileOutputStream outputStream = new FileOutputStream(fileName);
            FileChannel fileChannel = outputStream.getChannel();
            fileChannel.transferFrom(urlChannel, 0, headerSize);

            urlChannel.close();
        }
        catch (IOException e)
        {
            throw new RuntimeException("IO error during download from remote safetensors url: " + url);
        }
    }
}
