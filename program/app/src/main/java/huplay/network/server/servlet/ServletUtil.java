package huplay.network.server.servlet;

import huplay.file.FileUtil;

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