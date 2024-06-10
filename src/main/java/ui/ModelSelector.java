package ui;

import network.info.Models;

import java.io.*;
import java.util.*;

public class ModelSelector
{
    private final PrintStream OUT;
    private final Map<String, Models> allModels;

    public ModelSelector(PrintStream OUT, Map<String, Models> allModels)
    {
        this.OUT = OUT;
        this.allModels = allModels;
    }

    public String select() throws IOException
    {
        return selectModel(allModels, "");
    }

    private String selectModel(Map<String, Models> models, String path) throws IOException
    {
        // Order the models based on the order
        var modelOrderMap = new HashMap<Integer, String>();
        for (var model : models.entrySet())
        {
            modelOrderMap.put(model.getValue().getOrder(), model.getKey());
        }

        var modelOrders = new ArrayList<>(modelOrderMap.keySet());
        Collections.sort(modelOrders);

        List<String> modelList = new ArrayList<>();
        for (var modelOrder : modelOrders)
        {
            modelList.add(modelOrderMap.get(modelOrder));
        }

        // Display the list of models
        var maxIdLength = String.valueOf(models.size()).length();
        if (!path.isEmpty())
        {
            OUT.println(alignRight("0", maxIdLength) + ": ..");
        }

        int i = 1;
        for (var model : modelList)
        {
            var id = alignRight("" + i, maxIdLength);
            OUT.println(id + ": " + model);
            i++;
        }

        // Ask user to select (repeat at incorrect selection)
        var reader = new BufferedReader(new InputStreamReader(System.in));

        int choice;
        while (true)
        {
            OUT.print("\nPlease select: ");
            var text = reader.readLine();
            OUT.println();

            try
            {
                if (text.equals("x") || text.equals("X"))
                {
                    choice = -1;
                    break;
                }

                choice = Integer.parseInt(text);
                if ( (choice > 0 && choice <= i) || (!path.isEmpty() && choice == 0))
                {
                    break;
                }

                throw new Exception();
            }
            catch (Exception e)
            {
                OUT.println("Incorrect choice. (Press X to exit any time.)");
            }
        }

        if (choice == -1)
        {
            OUT.println("Bye!");
            System.exit(0);
        }
        else if (choice == 0)
        {
            // Go back one level
            var parentPath = getParentFolder(path);
            var parentModels = getModels(allModels, parentPath);
            return selectModel(parentModels, parentPath);
        }
        else
        {
            var modelName = modelList.get(choice - 1);
            var selectedModel = models.get(modelName);

            var newPath = path + (path.isEmpty() ? "" : "/") + modelName;
            if (selectedModel.getFolders() == null || selectedModel.getFolders().isEmpty())
            {
                return newPath;
            }
            else
            {
                return selectModel(selectedModel.getFolders(), newPath);
            }
        }

        return null;
    }

    private String alignRight(String text, int length)
    {
        if (text.length() >= length) return text;

        var pad = new char[length - text.length()];
        Arrays.fill(pad, ' ');
        return new String(pad) + text;
    }

    private String getParentFolder(String path)
    {
        var lastIndex = path.lastIndexOf("/");

        if (lastIndex == -1) return "";
        else return path.substring(0, lastIndex);
    }

    private Map<String, Models> getModels(Map<String, Models> models, String path)
    {
        if (path.isEmpty())
        {
            return models;
        }
        else
        {
            String[] splitPath = path.split("/");

            if (splitPath.length == 1)
            {
                return models.get(path).getFolders();
            }
            else
            {
                var nextModels = models.get(splitPath[0]).getFolders();
                var nextPath = path.substring(splitPath.length + 1);
                return getModels(nextModels, nextPath);
            }
        }
    }
}
