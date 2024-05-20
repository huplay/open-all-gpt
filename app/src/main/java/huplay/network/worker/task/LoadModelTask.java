package huplay.network.worker.task;

import huplay.config.Config;
import huplay.file.DownloadMissingFiles;
import huplay.file.SafetensorsReader;
import huplay.network.Address;
import huplay.network.message.toWorker.LoadModelMessage;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;
import huplay.transformer.TransformerType;
import huplay.ui.DownloadProgressBar;

import static huplay.AppNetworkWorker.OUT;
import static huplay.file.FileUtil.checkFiles;
import static huplay.network.worker.state.WorkerState.getWorkerState;

public class LoadModelTask implements Runnable
{
    private final LoadModelMessage request;
    private final Address server;

    public LoadModelTask(LoadModelMessage request, Address server)
    {
        this.request = request;
        this.server = server;
    }

    @Override
    public void run()
    {
        try
        {
            var modelId = request.getModelId();
            var modelConfig = request.getModelConfig();
            var taskUUID = request.getTaskUUID();
            var workSegment = request.getWorkSegment();
            var segmentType = workSegment.getWorkSegmentType();

            // Check necessary files
            var missingFiles = checkFiles(modelConfig, modelConfig.getDownloadPath());

            // Download the missing files
            var progressBar = new DownloadProgressBar(OUT);
            var downloader = new DownloadMissingFiles(progressBar);
            downloader.download(missingFiles, modelConfig, modelConfig.getDownloadPath());

            // Create the parameter reader
            var reader = new SafetensorsReader(modelConfig.getDownloadPath());

            Config config;
            if (modelConfig.getConfigOverride() != null)
            {
                // If the repo doesn't contain the config.json it is possible to include in the model.json
                // Use that if provided
                config = modelConfig.getConfigOverride();
            }
            else
            {
                config = Config.readConfig(null, modelConfig, null, reader);
            }

            var transformer = TransformerType.getTransformer(config.getTransformerType());

            // The tail workSegment isn't sent as load request, so we have to deal with the head and the layers only
            if (segmentType.hasHead())
            {
                transformer.init(config);
            }

            if (segmentType.hasLayer())
            {
                for (var decoderLayer : workSegment.getDecoderBlocks())
                {
                    transformer.initDecoderLayer(config, decoderLayer.getDecoderId(), decoderLayer.getBlockType());
                }
            }

            getWorkerState().activateModel(modelId, transformer);

            var message = new ModelLoadedMessage(taskUUID, modelId);
            message.send(server);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e); // TODO: Handle errors
        }
    }
}
