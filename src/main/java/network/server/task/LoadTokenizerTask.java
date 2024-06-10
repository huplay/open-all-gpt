package network.server.task;

import config.TokenizerConfig;
import parameters.download.DownloadMissingFiles;
import network.server.state.ServerState;
import tokenizer.TokenizerType;
import ui.DownloadProgressBar;

import static app.AppNetworkWorker.OUT;
import static parameters.FileUtil.checkFiles;

public class LoadTokenizerTask implements Runnable
{
    private final String taskUUID;
    private final String modelId;
    private final TokenizerConfig tokenizerConfig;
    private final String downloadPath;

    public LoadTokenizerTask(String taskUUID, String modelId, TokenizerConfig tokenizerConfig, String downloadPath)
    {
        this.taskUUID = taskUUID;
        this.modelId = modelId;
        this.tokenizerConfig = tokenizerConfig;
        this.downloadPath = downloadPath;
    }

    @Override
    public void run()
    {
        try
        {
            // Download the missing tokenizer files
            var missingFiles = checkFiles(tokenizerConfig, downloadPath);

            var progressBar = new DownloadProgressBar(OUT);
            var downloader = new DownloadMissingFiles(progressBar);
            downloader.download(missingFiles, tokenizerConfig, downloadPath);

            var tokenizer = TokenizerType.getTokenizer(tokenizerConfig);

            ServerState serverState = ServerState.getServerState();
            serverState.getPendingModels().get(modelId).setTokenizer(tokenizer);

            serverState.registerFinishedTask(modelId, taskUUID);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
}
