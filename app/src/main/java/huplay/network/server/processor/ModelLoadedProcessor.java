package huplay.network.server.processor;

import huplay.network.message.Acknowledge;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;

import static huplay.network.server.state.ServerState.getServerState;

public class ModelLoadedProcessor
{
    synchronized public static Acknowledge process(ModelLoadedMessage message)
    {
        var modelId = message.getModelId();
        System.out.println("ModelLoadedMessage received for " + modelId);

        getServerState().registerFinishedTask(modelId, message.getTaskUUID());

        return new Acknowledge();
    }
}
