package huplay.network.server.processor;

import huplay.network.message.toServer.fromClient.PollOpenModelRequest;
import huplay.network.message.toServer.fromClient.PollOpenModelResponse;
import huplay.network.server.task.LoadModelPlanTask;

import static huplay.network.server.state.ServerState.getServerState;

public class PollOpenModelProcessor
{
    public static PollOpenModelResponse process(PollOpenModelRequest message)
    {
        var modelId = message.getModelId();
        if (message.getAttempt() == 0) System.out.println("PollOpenModel received for " + modelId);

        boolean isReady = false;

        var serverState = getServerState();
        if (serverState.getPendingModels().containsKey(modelId))
        {
            // It is already in progress, do nothing
        }
        else if (serverState.getActiveModels().containsKey(modelId))
        {
            System.out.println("Model is ready (" + modelId + ")");
            isReady = true;
        }
        else
        {
            // This is the first time the model is requested, execute the load model plan task
            System.out.println("Execute load model plan task (" + modelId + ")");
            var task = new LoadModelPlanTask(modelId);
            new Thread(task).start();
        }

        return new PollOpenModelResponse(isReady);
    }
}
