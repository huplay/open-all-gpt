package network.server.servlet;

import parameters.FileUtil;

import java.io.File;

public class ServletUtil
{
    public static String getHtmlPage(String page)
    {
        var staticRoot = new File("static");
        File file = new File(staticRoot.getAbsolutePath() + "/html/" + page);

        return FileUtil.readTextFile(file.getAbsolutePath());
    }
}
