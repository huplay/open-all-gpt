package huplay.file;

public interface DownloadProgressHandler
{
    void showFile(String fileName, long size);

    void showProgressBar(long total, long actual, int length);
}
