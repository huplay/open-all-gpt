package network.server.state;

import network.Address;
import network.info.Models;
import network.info.WorkSegment;
import tokenizer.Tokenizer;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ServerState
{
    private static final ServerState SERVER_STATE = new ServerState();

    private final String configRoot;
    private final String downloadRoot;
    private Map<String, Models> models;

    private final Map<Address, WorkerInfo> workers = new HashMap<>();

    // Key: modelId
    private final Map<String, ModelState> activeModels = new HashMap<>();
    private final Map<String, ModelState> pendingModels = new HashMap<>();

    // Key: queryUUID
    private final Map<String, QueryState> finishedQueries = new HashMap<>();
    private final Map<String, QueryState> pendingQueries = new HashMap<>();

    // Key: workUUID
    private final Map<String, QueryState> pendingWorks = new HashMap<>();

    private ServerState()
    {
        File file = new File("models");
        this.configRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_MODELS_ROOT", file.getAbsolutePath());

        file = new File("download");
        this.downloadRoot = System.getenv().getOrDefault("OPEN_ALL_GPT_DOWNLOAD_ROOT", file.getAbsolutePath());
    }

    public static ServerState getServerState()
    {
        return SERVER_STATE;
    }

    public void addWorker(Address worker, long freeMemory)
    {
        workers.put(worker, new WorkerInfo(worker, freeMemory));
    }

    synchronized public ModelState addPendingModel(String modelId)
    {
        var serverState = getServerState();
        if (serverState.getActiveModels().containsKey(modelId))
        {
            // Double-check the model isn't active already
            return null;
        }

        var pendingModels = serverState.getPendingModels();
        var pendingModel = pendingModels.get(modelId);
        if (pendingModel == null)
        {
            // There's no loading state, this is the first time the model is requested: create new loading state
            ModelState modelState = new ModelState();
            pendingModels.put(modelId, modelState);

            return modelState;
        }

        // The loading state was already created, the loading is in progress
        return null;
    }

    public void registerFinishedTask(String modelId, String taskUUID)
    {
        var pendingTasks = pendingModels.get(modelId).getPendingTasks();
        pendingTasks.remove(taskUUID);

        if (pendingTasks.isEmpty())
        {
            activateModel(modelId);
        }
    }

    synchronized
    public void activateModel(String modelId)
    {
        activeModels.put(modelId, pendingModels.get(modelId));
        pendingModels.remove(modelId);
    }

    public void addPendingQuery(String queryUUID, QueryState queryState)
    {
        pendingQueries.put(queryUUID, queryState);
    }

    public void addPendingWork(String workUUID, QueryState queryState)
    {
        pendingWorks.put(workUUID, queryState);
    }

    public void removePendingWork(String workUUID)
    {
        pendingWorks.remove(workUUID);
    }

    public QueryState getPendingQueryState(String workUUID)
    {
        return pendingWorks.get(workUUID);
    }

    synchronized
    public void activateQuery(String queryUUID)
    {
        finishedQueries.put(queryUUID, pendingQueries.get(queryUUID));
        pendingModels.remove(queryUUID);
    }

    public List<WorkSegment> getWorkSegments(String modelId)
    {
        return activeModels.get(modelId).getWorkSegments();
    }

    public Tokenizer getTokenizer(String modelId)
    {
        return activeModels.get(modelId).getTokenizer();
    }

    // Getters
    public String getConfigRoot() {return configRoot;}
    public String getDownloadRoot() {return downloadRoot;}
    public Map<String, Models> getModels() {return models;}
    public Map<Address, WorkerInfo> getWorkers() {return workers;}
    public Map<String, ModelState> getActiveModels() {return activeModels;}
    public Map<String, ModelState> getPendingModels() {return pendingModels;}
    public Map<String, QueryState> getFinishedQueries() {return finishedQueries;}
    public Map<String, QueryState> getPendingQueries() {return pendingQueries;}

    // Setter
    public void setModels(Map<String, Models> models) {this.models = models;}
}
