package network.server.processor;

import network.message.Acknowledge;
import network.message.toServer.fromWorker.ModelLoadedMessage;

import static network.server.state.ServerState.getServerState;

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
