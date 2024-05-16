package huplay;

import huplay.config.*;
import huplay.file.DownloadMissingFiles;
import huplay.file.SafetensorsReader;
import huplay.transformer.TransformerType;
import huplay.ui.DownloadProgressBar;
import huplay.util.Util;

import java.io.*;
import java.util.*;

import static huplay.AppStandaloneMain.*;
import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.Logo.logo;
import static huplay.ui.TextUtil.toCenter;
import static java.lang.Math.round;

public class AppStandaloneLauncher
{
    public static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static void main(String... args)
    {
        try
        {
            logo(OUT,"Open All GPT", "CWgY-CWY-bgW", 'W');
            OUT.println(toCenter("Standalone Launcher\n", 60));

            OUT.println(toCenter("Util: " + UTIL.getUtilName() + "\n", 60));

            new AppStandaloneLauncher().start(args);
        }
        catch (Exception e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
    }

    private void start(String... args) throws Exception
    {
        // Read arguments
        var arguments = Arguments.read(args);

        // If the model isn't specified, allow the user to select it
        if (arguments.getConfigPath() == null)
        {
            selectModel(arguments);
        }

        // Read the modelConfig of the selected model
        var modelConfig = ModelConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        // Read the modelConfig of the selected model
        var tokenizerConfig = TokenizerConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        // Check necessary files and download the missing
        var missingFiles = checkFiles(modelConfig, arguments.getModelPath());
        var progressBar = new DownloadProgressBar(OUT);
        var downloader = new DownloadMissingFiles(progressBar);
        downloader.download(missingFiles, modelConfig, arguments.getModelPath());

        // Check necessary files for tokenizer
        missingFiles = checkFiles(tokenizerConfig, arguments.getModelPath());
        downloader.download(missingFiles, modelConfig, arguments.getModelPath());

        var reader = new SafetensorsReader(arguments.getModelPath());

        // Read the config (first look into the model folder, second to the config folder (maybe it's different)
        var config = Config.readConfig(arguments, modelConfig, tokenizerConfig, reader);

        if (arguments.isCalculationOnly())
        {
            // Calculation only. Display config, parameter size
            config.setCalculationOnly(true);

            var transformer = TransformerType.getTransformer(config.getTransformerType());
            transformer.init(config);
            transformer.initDecoders();

            displayConfig(config, transformer.getParameterSize());
        }
        else
        {
            // Determine memory requirement
            var memorySize = determineMemoryRequirement(config);

            try
            {
                var userDir = System.getProperty("user.dir").replace('\\', '/');

                // Open the main app to launch the model
                var command = "java" +
                                " -Xmx" + memorySize + "m -Xms" + memorySize + "m" +
                                " -cp " + userDir + "/app/target/open-all-gpt.jar" +
                                " huplay.AppStandaloneMain" +
                                " \"" + arguments.getModelId() + "\"" +
                                " -max=" + config.getLengthLimit() +
                                " -topK=" + config.getTopK();

                OUT.println("Command:\n" + command + "\n");
                Runtime.getRuntime().exec("cmd /k start cmd /c " + command); // TODO: Deprecated
            }
            catch (IOException e)
            {
                OUT.println("Error launching the main app: " + e.getMessage());
            }
        }
    }

    private void selectModel(Arguments arguments) throws Exception
    {
        var configRoot = arguments.getConfigRoot();
        var configPath = selectModel(configRoot, configRoot);
        var relativePath = configPath.substring(configRoot.length() + 1);

        arguments.setModelId(relativePath);
    }

    private String selectModel(String path, String configRoot) throws IOException
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
                return selectModel(getParentFolder(path), configRoot);
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
                return selectModel(newPath, configRoot);
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

    private static int determineMemoryRequirement(Config config)
    {
        // First, use the requested memory size (if exists)
        var memorySize = config.getRequestedMemorySize();
        if (memorySize == 0)
        {
            // Second, use the configured total memory size (if exists)
            memorySize = config.getMemorySize();
            if (memorySize == null || memorySize == 0)
            {
                // Third, calculate the required memory
                config.setCalculationOnly(true);

                var transformer = TransformerType.getTransformer(config.getTransformerType());
                transformer.init(config);
                transformer.initDecoders();

                var parameterMemorySize = round((float) transformer.getParameterSize() / 1000 / 1000 * 4);

                memorySize = parameterMemorySize + 2048;
            }
        }

        return memorySize;
    }

    public static List<String> checkFiles(RepoConfig modelConfig, String modelPath)
    {
        var missingFiles = new ArrayList<String>();

        for (var fileName : modelConfig.getFiles())
        {
            var path = modelPath + "/" + fileName;

            var file = new File(path);
            if (!file.exists())
            {
                missingFiles.add(fileName);
            }
        }

        return missingFiles;
    }
}
