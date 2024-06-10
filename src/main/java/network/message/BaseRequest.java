package network.message;

import com.fasterxml.jackson.databind.ObjectMapper;
import network.Address;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public abstract class BaseRequest<T extends BaseResponse> extends BaseMessage
{
    public T send(Address target) throws IOException
    {
        var messageJson = new ObjectMapper().writeValueAsString(this);

        var connection = sendRequest(target, messageJson);

        var responseCode = connection.getResponseCode();
        if (responseCode >= 100 && responseCode <= 399)
        {
            var reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));

            var builder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null)
            {
                builder.append(line);
            }

            var response = new ObjectMapper().readValue(builder.toString(), BaseMessage.class);

            return (T) response;
        }
        else
        {
            System.out.println("Error during sending request: " + responseCode + " " + connection.getErrorStream().toString());
            //BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getErrorStream()));
            // TODO: Handle error
            return null;
        }
    }

    private static HttpURLConnection sendRequest(Address target, String messageJson) throws IOException
    {
        URL url = new URL(target.getURL()); // TODO: Deprecated

        var connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setFixedLengthStreamingMode(messageJson.length());
        connection.setRequestProperty("Content-Type", "application/json");
        connection.connect();

        try (var writer = new OutputStreamWriter(connection.getOutputStream(), StandardCharsets.UTF_8))
        {
            writer.write(messageJson);
            writer.flush();
        }

        return connection;
    }
}
