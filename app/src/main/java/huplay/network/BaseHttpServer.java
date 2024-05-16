package huplay.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import huplay.network.message.BaseMessage;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public abstract class BaseHttpServer implements HttpHandler
{
    private final ObjectMapper objectMapper = new ObjectMapper();

    protected abstract BaseMessage handlePostMessage(BaseMessage message);

    protected abstract String handleGetMessage(String message);

    @Override
    public void handle(HttpExchange received)
    {
        received.getRequestMethod();

        try
        {
            var method = received.getRequestMethod();
            if (method.equals("POST"))
            {
                var requestString = getRequestBody(received);
                var requestJson = objectMapper.readValue(requestString, BaseMessage.class);

                var responseJson = handlePostMessage(requestJson);
                var responseString = objectMapper.writeValueAsString(responseJson);

                sendResponse(received, responseString, 200);
            }
            else if (method.equals("GET"))
            {
                var responseHTML = handleGetMessage("");

                sendResponse(received, responseHTML, 200);
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.out.println("ERROR during handling request: " + e.getMessage());
            sendResponse(received, "ERROR during handling request" + e.getMessage(), 500);
        }
    }

    private String getRequestBody(HttpExchange message) throws IOException
    {
        var inputStream = new BufferedInputStream(message.getRequestBody());
        var outputStream = new ByteArrayOutputStream();
        for (var result = inputStream.read(); result != -1; result = inputStream.read())
        {
            outputStream.write((byte) result);
        }

        return outputStream.toString(StandardCharsets.UTF_8);
    }

    private void sendResponse(HttpExchange received, String response, int code)
    {
        try
        {
            received.sendResponseHeaders(code, response.length());

            var outputStream = received.getResponseBody();
            outputStream.write(response.getBytes());
            outputStream.close();
        }
        catch (IOException e)
        {
            System.out.println("ERROR during sending response: " + e.getMessage());
        }
    }
}
