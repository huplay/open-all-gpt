package huplay;

import com.sun.net.httpserver.HttpServer;
import huplay.network.Endpoint;
import huplay.network.server.ServerListener;

import java.io.IOException;
import java.io.PrintStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;

import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.ConsoleUtil.intInput;
import static huplay.ui.Logo.logoCenter;
import static huplay.ui.TextUtil.*;

public class AppNetworkServer
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            clearScreen(OUT, 'X');
            logoCenter(OUT,"Open All GPT", "------------", '-', 60);
            logoCenter(OUT,"Server", "------------", '-', 60);

            new AppNetworkServer().start(args);
        }
        catch (Exception e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
    }

    private void start(String... args) throws IOException
    {
        // Read port
        Integer port = null;
        if (args != null && args.length > 0)
        {
            try
            {
                port = Integer.parseInt(args[0]);
            }
            catch (NumberFormatException e)
            {
                OUT.println("WARNING: Cannot read parameter as port: " + args[0]);
            }
        }

        if (port == null)
        {
            port = intInput(OUT, toCenter("Server port: ", 54), "Wrong number");
        }

        var server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext(Endpoint.SERVER.getContext(), new ServerListener());
        server.setExecutor(null); // creates a default executor
        server.start();

        OUT.println(toCenter(InetAddress.getLocalHost().getHostAddress() + " listening on port: " + port + "\n", 60));
    }
}
