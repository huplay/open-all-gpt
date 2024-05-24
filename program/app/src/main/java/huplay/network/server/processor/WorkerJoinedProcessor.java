package huplay.network.server.processor;

import huplay.network.message.Acknowledge;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;

import static huplay.network.server.state.ServerState.getServerState;

public class WorkerJoinedProcessor
{
    public static Acknowledge process(WorkerJoinedMessage message)
    {
        var workerAddress = message.getWorkerAddress();
        System.out.println("WorkerJoinMessage received from " + workerAddress);

        getServerState().addWorker(workerAddress, message.getFreeMemory());

        return new Acknowledge();
    }
}
