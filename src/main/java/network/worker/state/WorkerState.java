package network.worker.state;

import transformer.BaseTransformer;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class WorkerState
{
    private static final WorkerState WORKER_STATE = new WorkerState();

    private final String configRoot;

    // Key: modelId
    private final Map<String, BaseTransformer> activeModels = new HashMap<>();
    private final Set<String> pendingModels = new HashSet<>();

    private WorkerState()
    {
        var file = new File("models");
        this.configRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_MODELS_ROOT", file.getAbsolutePath());
    }

    public static WorkerState getWorkerState()
    {
        return WORKER_STATE;
    }

    // Getters
    public String getConfigRoot() {return configRoot;}
    public Set<String> getPendingModels() {return pendingModels;}

    synchronized
    public void activateModel(String modelId, BaseTransformer transformer)
    {
        activeModels.put(modelId, transformer);
        pendingModels.remove(modelId);
    }

    public BaseTransformer getActiveModel(String modelId)
    {
        return activeModels.get(modelId);
    }
}
