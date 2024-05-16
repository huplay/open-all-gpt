package huplay.file;

import huplay.config.RepoConfig;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static huplay.file.DownloadUtil.determineDownloadUrl;

public class DownloadTask implements Runnable
{
    private static final int BATCH_SIZE = 5120;

    private final String fileName;
    private final String path;
    private final URL url;

    private boolean isInProgress = true;
    private final long size;
    private long pieces;
    private long position = 0;
    private boolean isOk = true;

    public DownloadTask(RepoConfig repoConfig, String fileName, String path) throws Exception
    {
        this.fileName = fileName;
        this.path = path;
        this.url = new URL(determineDownloadUrl(repoConfig, fileName)); // TODO: Deprecated
        this.size = getSize(url);

        this.pieces = Math.floorDiv(size, BATCH_SIZE);
        if (size % BATCH_SIZE > 0) pieces++;
    }

    @Override
    public void run()
    {
        try (var outputStream = new FileOutputStream(path + "/" + fileName))
        {
            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());
            FileChannel fileChannel = outputStream.getChannel();

            long startPos = 0;
            for (position = 0; position < pieces; position++)
            {
                fileChannel.transferFrom(urlChannel, startPos, BATCH_SIZE);
                startPos += BATCH_SIZE;
            }

            urlChannel.close();
        }
        catch (IOException e)
        {
            isOk = false;
            throw new RuntimeException("IO error during file download: " + fileName + " error: " + e);
        }

        isInProgress = false;
    }

    private long getSize(URL url) throws IOException
    {
        var urlConnection = url.openConnection();
        urlConnection.connect();
        return urlConnection.getContentLengthLong();
    }

    // Getters
    public boolean isInProgress() {return isInProgress;}
    public boolean isOk() {return isOk;}
    public long getSize() {return size;}
    public long getPieces() {return pieces;}
    public long getPosition() {return position;}
}
