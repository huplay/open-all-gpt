package huplay.file.download;

public interface DownloadProgressHandler
{
    void showFile(String fileName, long size);

    void showProgressBar(long total, long actual, int length);
}
