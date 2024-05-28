package huplay.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import huplay.parameters.FileUtil;
import huplay.network.message.BaseMessage;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;

public abstract class BaseHttpServer implements HttpHandler
{
    private final ObjectMapper objectMapper = new ObjectMapper();

    protected abstract BaseMessage handlePostMessage(BaseMessage message);

    protected abstract String handleGetMessage(String context, String path, String query);

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

                sendResponse(received, responseString);
            }
            else if (method.equals("GET"))
            {
                var uri = received.getRequestURI();

                var domain = Endpoint.SERVER.getDomain();
                var uriPath = uri.getPath();

                // Sanity check of the url
                if (uriPath == null
                        || uriPath.length() < domain.length()
                        || !uriPath.startsWith(domain)
                        || (uriPath.length() > domain.length() && uriPath.charAt(domain.length()) != '/'))
                {
                    // Something wrong, we should not be here
                    sendError(received, "Open All GPT - 404 Page Not Found", 404);
                }
                else
                {
                    // Remove the domain part from the url
                    var fullPath = uriPath.substring(Endpoint.SERVER.getDomain().length());

                    // Clean and split the url into context and path
                    var pathList = new ArrayList<String>();
                    for (var part : fullPath.split("/"))
                    {
                        if (!part.isEmpty()) pathList.add(part);
                    }

                    var context = pathList.isEmpty() ? "/" : pathList.getFirst();
                    if (!pathList.isEmpty()) pathList.removeFirst();
                    var path = String.join("/", pathList);

                    if (context.equals("static"))
                    {
                        sendStaticResponse(received, path);
                    }
                    else
                    {
                        var responseHTML = handleGetMessage(context, path, uri.getQuery());

                        sendResponse(received, responseHTML);
                    }
                }
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

    private void sendResponse(HttpExchange received, String response)
    {
        sendResponse(received, response, 200);
    }

    private void sendError(HttpExchange received, String response, int code)
    {
        sendResponse(received, response, code);
    }

    private void sendStaticResponse(HttpExchange received, String path)
    {
        var staticRoot = new File("static");
        File file = new File(staticRoot.getAbsolutePath() + "/" + path);

        if (!file.exists() || !file.isFile())
        {
            sendError(received, "Open All GPT - 404 Content Not Found", 404);
        }
        else if (path.startsWith("image/"))
        {
            try
            {
                File image = new File("static/" + path);

                if (!image.exists() && !image.isFile())
                {
                    sendError(received, "Open All GPT - 404 Content Not Found", 404);
                }
                else
                {
                    int size = (int) image.length();
                    received.sendResponseHeaders(200, size);
                    OutputStream outputStream = received.getResponseBody();
                    Files.copy(file.toPath(), outputStream);
                    outputStream.close();
                }
            }
            catch (IOException e)
            {
                sendError(received, "Open All GPT - 404 Content Not Found", 404);
            }
        }
        else
        {
            var content = FileUtil.readTextFile(file.getAbsolutePath());

            sendResponse(received, content);
        }
    }
}
