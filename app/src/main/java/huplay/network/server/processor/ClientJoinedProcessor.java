package huplay.network.server.processor;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.network.info.Models;
import huplay.network.message.toServer.fromClient.ClientJoinedResponse;

import java.util.Map;

import static huplay.file.FileUtil.readTextFile;
import static huplay.network.server.state.ServerState.getServerState;

public class ClientJoinedProcessor
{
    public static ClientJoinedResponse process()
    {
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
