package huplay.ui;

import java.io.*;
import java.util.*;

public class ModelSelector
{
    public static String selectModel(PrintStream OUT, String configRoot) throws Exception
    {
        var configPath = selectModel(OUT, configRoot, configRoot);
        return configPath.substring(configRoot.length() + 1);
    }

    private static String selectModel(PrintStream OUT, String path, String configRoot) throws IOException
    {
        var fileList = new File(path).listFiles();

        if (fileList != null)
        {
            var files = Arrays.asList(fileList);
            Collections.sort(files);

            // Find model.properties
            for (var file : files)
            {
                if (file.isFile() && file.getName().equals("model.json"))
                {
                    return path;
                }
            }

            // Find directories
            var directories = new LinkedHashMap<Integer, String>();
            var i = 1;
            var j = -1;
            var length = 1;
            for (var file : files)
            {
                var name = file.getName();
                if (file.isDirectory())
                {
                    if (isEnabled(name))
                    {
                        directories.put(i, name);
                        length = String.valueOf(i).length();
                        i++;
                    }
                    else
                    {
                        directories.put(j, name);
                        j--;
                    }
                }
            }

            if (directories.isEmpty())
            {
                // Go back a level if there's no model here and no subfolders
                OUT.println("There is no model in the selected folder.");
                return selectModel(OUT, getParentFolder(path), configRoot);
            }
            else
            {
                if (!path.equals(configRoot))
                {
                    OUT.println(alignRight("0", length) + ": ..");
                }

                // Display the list of directories
                for (Map.Entry<Integer, String> entry : directories.entrySet())
                {
                    var key = entry.getKey();
                    var id = alignRight((key > 0) ? entry.getKey().toString() : "-", length);

                    var displayName = getDisplayName(entry.getValue());

                    OUT.println(id + ": " + displayName);
                    i++;
                }

                // Ask user to select (repeat at incorrect selection)
                var reader = new BufferedReader(new InputStreamReader(System.in));

                int choice;
                while (true)
                {
                    OUT.print("Please select: ");
                    var text = reader.readLine();

                    try
                    {
                        if (text.equals("x") || text.equals("X"))
                        {
                            choice = -1;
                            break;
                        }

                        choice = Integer.parseInt(text);
                        if ( (choice > 0 && choice <= directories.size()) || (!path.equals(configRoot) && choice == 0))
                        {
                            break;
                        }

                        OUT.println("Incorrect choice. (Press X to exit any time.)");
                    }
                    catch (Exception e)
                    {
                        OUT.println("Incorrect choice. (Press X to exit any time.)");
                    }
                }

                var newPath = "";
                if (choice == -1)
                {
                    OUT.println("Bye!");
                    System.exit(0);
                }
                else if (choice == 0)
                {
                    newPath = getParentFolder(path);
                }
                else
                {
                    newPath = path + "/" + directories.get(choice);
                }

                OUT.println();
                return selectModel(OUT, newPath, configRoot);
            }
        }

        OUT.println("There are no configured models.");
        OUT.println("Bye!");
        System.exit(0);
        return null;
    }

    private static boolean isEnabled(String name)
    {
        if (name.startsWith("("))
        {
            // Remove the bracketed order from the name
            var closing = name.indexOf(")");

            return closing <= 2 || name.charAt(closing - 2) != '-' || name.charAt(closing - 1) != '-';
        }

        return true;
    }

    private static String getDisplayName(String name)
    {
        if (name.startsWith("("))
        {
            // Remove the bracketed order from the name
            var closing = name.indexOf(")");
            if (closing > 0) name = name.substring(closing + 1);
        }

        return name;
    }

    private static String alignRight(String text, int length)
    {
        if (text.length() >= length) return text;

        var pad = new char[length - text.length()];
        Arrays.fill(pad, ' ');
        return new String(pad) + text;
    }

    private static String getParentFolder(String path)
    {
        var lastIndex = path.lastIndexOf("/");
        return path.substring(0, lastIndex);
    }
}
