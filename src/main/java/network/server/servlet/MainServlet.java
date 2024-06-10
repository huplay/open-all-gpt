package network.server.servlet;

public class MainServlet
{
    public static String get(String path, String query)
    {
        return ServletUtil.getHtmlPage("main.html");
    }
}
