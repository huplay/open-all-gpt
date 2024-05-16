package huplay.network.server.task;

import huplay.config.BlockType;
import huplay.config.Config;
import huplay.config.ModelConfig;
import huplay.config.TokenizerConfig;
import huplay.file.DownloadSafetensorsHeader;
import huplay.file.SafetensorsReader;
import huplay.network.info.DecoderBlock;
import huplay.network.info.DecoderBlockType;
import huplay.network.message.toWorker.LoadModelRequest;
import huplay.network.info.WorkSegment;
import huplay.network.info.WorkSegmentType;
import huplay.network.server.state.ModelState;
import huplay.network.server.state.WorkerInfo;
import huplay.transformer.TransformerType;

import java.io.IOException;
import java.util.*;

import static huplay.config.BlockType.*;
import static huplay.file.DownloadUtil.checkHeaderFiles;
import static huplay.network.server.state.ServerState.getServerState;
import static java.lang.Math.round;

public class LoadModelPlanTask implements Runnable
{
    private final String modelId;

    public LoadModelPlanTask(String modelId)
    {
        this.modelId = modelId;
    }

    @Override
    public void run()
    {
        var serverState = getServerState();
        var workers = new ArrayList<>(serverState.getWorkers().values());

        if (workers.isEmpty())
        {
            // TODO: Raise error
        }
        else
        {
            ModelState modelState = getServerState().addPendingModel(modelId);
            if (modelState != null) // Returns not null if the model was requested the first time, so we have to load it
            {
                try
                {
                    // Read the model.json of the model
                    var modelConfigPath = getServerState().getConfigRoot() + "/" + modelId;
                    var downloadPath = getServerState().getDownloadRoot() + "/" + modelId;
                    var modelConfig = ModelConfig.read(modelConfigPath, downloadPath);

                    // Determine required memory for main/attention/neural net layers (calculate or read from config)
                    var memoryRequirements = determineRequiredMemory(modelConfig, downloadPath);

                    // Split the work for workers
                    var workSegments = splitWork(workers, memoryRequirements);

                    // Create the load requests and register those as pending (do not send, wait until all registered)
                    var loadRequests = createLoadRequests(modelConfig, modelState, workSegments);

                    // Download the tokenizer files if missing
                    // This is also a pending task, but we can execute now because this is the last
                    startDownLoadTokenizerFilesTask(modelConfigPath, downloadPath, modelState);

                    // Send the model load requests to workers
                    for (var loadRequest : loadRequests)
                    {
                        var worker = loadRequest.getWorkSegment().getWorker();
                        loadRequest.send(worker);
                    }

                    // If we have multiple segments add a last segment as TAIL_ONLY by the first worker
                    // It is added after sending the load requests, to avoid sending two requests to the same worker
                    if (workSegments.size() > 1)
                    {
                        workSegments.add(new WorkSegment(workSegments.getFirst().getWorker()));
                        workSegments.getLast().setWorkSegmentType(WorkSegmentType.TAIL_ONLY);
                    }

                    // Store the workSegments, it will be used at sending the actual work tasks
                    modelState.setWorkSegments(workSegments);
                }
                catch (IOException e)
                {
                    // TODO: Raise
                    throw new RuntimeException("Error during sending load message", e);
                }
            }
        }
    }

    private void downloadHeaders(ModelConfig modelConfig, String downloadPath)
    {
        var headerPath = downloadPath + "/header";
        if (!checkHeaderFiles(downloadPath, headerPath))
        {
            for (String fileName : modelConfig.getFiles())
            {
                if (fileName.endsWith("safetensors"))
                {
                    DownloadSafetensorsHeader.downloadHeader(modelConfig, headerPath, fileName);
                }
            }
        }
    }

    private Map<BlockType, Long> determineRequiredMemory(ModelConfig modelConfig, String downloadPath)
    {
        var requirements = new HashMap<BlockType, Long>(3);

        var configuredRequirements = modelConfig.getMemorySizes();
        if (configuredRequirements != null && !configuredRequirements.isEmpty())
        {
            configuredRequirements.forEach((key, value) -> requirements.put(key, value * 1024L * 1024L));
        }
        else
        {
            // Download the safetensors headers if missing
            downloadHeaders(modelConfig, downloadPath);

            var reader = new SafetensorsReader(downloadPath);
            var config = Config.readConfig(null, modelConfig, null, reader);
            config.setCalculationOnly(true);

            var transformer = TransformerType.getTransformer(modelConfig.getTransformerType());

            transformer.init(config);
            long mainSize = transformer.getParameterByteSize();
            requirements.put(MAIN, mainSize);

            transformer.initDecoderLayer(config, 0, DecoderBlockType.ATTENTION_LAYER);
            long attentionSize = transformer.getParameterByteSize() - mainSize;
            requirements.put(ATTENTION_LAYER, attentionSize);

            transformer.initDecoderLayer(config, 0, DecoderBlockType.NEURAL_NET_LAYER);
            long neuralNetSize = transformer.getParameterByteSize() - mainSize - attentionSize;
            requirements.put(NEURAL_NET_LAYER, neuralNetSize);
        }

        return requirements;
    }

    private List<WorkSegment> splitWork(List<WorkerInfo> workers, Map<BlockType, Long> memoryRequirements)
    {
        var workSegments = new ArrayList<WorkSegment>();

        // TODO: This isn't exact measurement, fine-grain later
        float scale = 1.2f;

        long mainBlockRequirement = round(scale * memoryRequirements.get(MAIN)); // 50000 * 1024 * 4 * 2;
        long attentionRequirement = round(scale * memoryRequirements.get(ATTENTION_LAYER)); // worker.getFreeMemory() / 3 / 12;
        long neuralNetRequirement = round(scale * memoryRequirements.get(NEURAL_NET_LAYER)); // attentionRequirement

        // Order the workers by free memory
        workers.sort(Collections.reverseOrder());
        var workerIterator = workers.iterator();

        // Add the first worker doing the head
        var worker = workerIterator.next();
        workSegments.add(new WorkSegment(worker.getAddress()));

        long freeMemory = worker.getFreeMemory() - mainBlockRequirement;

        for (int i = 0; i < 12; i++) // TODO: Use modelSize
        {
            if (freeMemory < attentionRequirement)
            {
                if (!workerIterator.hasNext())
                {
                    // TODO: raise error, we don't have enough workers
                }
                else
                {
                    // Not enough memory for the worker to pick up a new segment, move to the next worker
                    worker = workerIterator.next();
                    workSegments.add(new WorkSegment(worker.getAddress()));
                    freeMemory = worker.getFreeMemory();
                }
            }
            freeMemory -= attentionRequirement;

            workSegments.getLast().addDecoderBlock(new DecoderBlock(DecoderBlockType.ATTENTION_LAYER, i));

            if (freeMemory < neuralNetRequirement)
            {
                if (!workerIterator.hasNext())
                {
                    // TODO: raise error, we don't have enough workers
                }
                else
                {
                    // Not enough memory for the worker to pick up a new segment, move to the next worker
                    worker = workerIterator.next();
                    workSegments.add(new WorkSegment(worker.getAddress()));
                    freeMemory = worker.getFreeMemory();
                }
            }
            freeMemory -= neuralNetRequirement;

            workSegments.getLast().addDecoderBlock(new DecoderBlock(DecoderBlockType.NEURAL_NET_LAYER, i));
        }

        markWorkSegmentTypes(workSegments);

        return workSegments;
    }

    private void markWorkSegmentTypes(List<WorkSegment> workSegments)
    {
        // Set the workSegmentType
        if (workSegments.size() == 1)
        {
            // If we have only one worker: FULL
            workSegments.getFirst().setWorkSegmentType(WorkSegmentType.FULL);
        }
        else
        {
            // The first worker has to do the HEAD and TAIL. Set the HEAD:
            var firstSegment = workSegments.getFirst();
            if (firstSegment.getDecoderBlocks().isEmpty())
            {
                firstSegment.setWorkSegmentType(WorkSegmentType.HEAD_ONLY);
            }
            else
            {
                firstSegment.setWorkSegmentType(WorkSegmentType.HEAD_AND_LAYERS);
            }

            // Mark the rest of the workers as LAYERS_ONLY
            for (int i = 1; i < workSegments.size(); i++)
            {
                workSegments.get(i).setWorkSegmentType(WorkSegmentType.LAYERS_ONLY);
            }
        }
    }

    private List<LoadModelRequest> createLoadRequests(ModelConfig modelConfig, ModelState modelState, List<WorkSegment> workSegments)
    {
        // Collect the load requests to the workers
        // We will send the request later, after all tasks are added as pending task,
        // to make sure none of it will finish before another is registered.
        var loadRequests = new ArrayList<LoadModelRequest>();

        for (WorkSegment workSegment : workSegments)
        {
            // Send a LoadModelRequest message to the worker
            var loadTaskUUID = UUID.randomUUID().toString();
            modelState.getPendingTasks().put(loadTaskUUID, "LOAD request" + workSegment.getWorker());

            loadRequests.add(new LoadModelRequest(loadTaskUUID, modelId, modelConfig, workSegment));
        }

        return loadRequests;
    }

    private void startDownLoadTokenizerFilesTask(String modelConfigPath, String downloadPath, ModelState modelState)
    {
        // Start a task to download tokenizer files (if missing)
        var tokenizerConfig = TokenizerConfig.read(modelConfigPath, downloadPath);
        var tokenizerTaskUUID = UUID.randomUUID().toString();
        modelState.getPendingTasks().put(tokenizerTaskUUID, "LOAD tokenizer request");
        var tokenizerTask = new LoadTokenizerTask(tokenizerTaskUUID, modelId, tokenizerConfig, downloadPath);
        new Thread(tokenizerTask).start();
    }
}
