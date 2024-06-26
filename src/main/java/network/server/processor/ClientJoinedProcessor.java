package network.server.processor;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import network.info.Models;
import network.message.toServer.fromClient.ClientJoinedResponse;

import java.util.Map;

import static parameters.FileUtil.readTextFile;
import static network.server.state.ServerState.getServerState;

public class ClientJoinedProcessor
{
    public static ClientJoinedResponse process()
    {
        System.out.println("ClientJoinedResponse received");
        var serverState = getServerState();
        var models = serverState.getModels();

        if (models == null)
        {
            // Read the models at first request
            try
            {
                var modelsJson = readTextFile(serverState.getConfigRoot() + "/models.json");
                var typeRef = new TypeReference<Map<String, Models>>(){};
                models = new ObjectMapper().readValue(modelsJson, typeRef);

                serverState.setModels(models);
            }
            catch (JsonProcessingException e)
            {
                throw new RuntimeException("Error reading the models.json file. " + e.getMessage());
            }
        }

        return new ClientJoinedResponse(models);
    }
}
