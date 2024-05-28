package huplay.network.server.task;

import huplay.config.TokenizerConfig;
import huplay.parameters.download.DownloadMissingFiles;
import huplay.network.server.state.ServerState;
import huplay.tokenizer.TokenizerType;
import huplay.ui.DownloadProgressBar;

import static huplay.AppNetworkWorker.OUT;
import static huplay.parameters.FileUtil.checkFiles;
import static huplay.network.server.state.ServerState.getServerState;

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

            ServerState serverState = getServerState();
            serverState.getPendingModels().get(modelId).setTokenizer(tokenizer);

            serverState.registerFinishedTask(modelId, taskUUID);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
}
