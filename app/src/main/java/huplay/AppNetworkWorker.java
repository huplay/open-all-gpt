package huplay;

import com.sun.net.httpserver.HttpServer;
import huplay.network.Address;
import huplay.network.Endpoint;
import huplay.network.NetworkSettings;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;
import huplay.network.worker.WorkerListener;

import java.io.PrintStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;

import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.Logo.logoCenter;
import static huplay.ui.TextUtil.*;

public class AppNetworkWorker
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            clearScreen(OUT, 'B');
            logoCenter(OUT,"Open All GPT", "WWWW-WWW-WWW", 'W', 60);
            logoCenter(OUT,"Worker", "WWWWWW", 'W', 60);

            // Read arguments
            var arguments = NetworkSettings.read(OUT, args, true);

            // Determine self address
            var host = InetAddress.getLocalHost().getHostAddress();
            var self = new Address(Endpoint.WORKER, host, arguments.getSelfPort());

            // Determine server address
            var server = new Address(Endpoint.SERVER, arguments.getServerHost(), arguments.getServerPort());

            // Start Worker
            var httpServer = HttpServer.create(new InetSocketAddress(arguments.getSelfPort()), 0);
            httpServer.createContext(Endpoint.WORKER.getContext(), new WorkerListener(server));
            httpServer.setExecutor(null); // creates a default executor
            httpServer.start();

            OUT.println(toCenter(InetAddress.getLocalHost().getHostAddress() + " listening on port: " + arguments.getSelfPort() + "\n", 60));

            // Send WorkerJoined message to server
            var freeMemory = Runtime.getRuntime().freeMemory();
            var workerJoinedMessage = new WorkerJoinedMessage(self, freeMemory);

            workerJoinedMessage.send(server);

            // Stay open console
            while (true)
            {
                Thread.sleep(10000);
            }
        }
        catch (Exception e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
    }
}
