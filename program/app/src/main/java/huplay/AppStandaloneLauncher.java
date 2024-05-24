package huplay;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.config.*;
import huplay.file.download.DownloadMissingFiles;
import huplay.file.safetensors.SafetensorsReader;
import huplay.network.info.Models;
import huplay.transformer.TransformerType;
import huplay.ui.DownloadProgressBar;
import huplay.ui.ModelSelector;

import java.io.*;
import java.util.Map;

import static huplay.AppStandaloneMain.*;
import static huplay.file.FileUtil.checkFiles;
import static huplay.file.FileUtil.readTextFile;
import static huplay.MathUtilProvider.MATH;
import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.Logo.logo;
import static huplay.ui.TextUtil.toCenter;
import static java.lang.Math.round;

public class AppStandaloneLauncher
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            logo(OUT,"Open All GPT", "CWgY-CWY-bgW", 'W');
            OUT.println(toCenter("Standalone Launcher\n", 60));

            OUT.println(toCenter("Math module: " + MATH.getMathProviderName() + "\n", 60));

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
            selectModel(OUT, arguments);
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
                                " -cp " + userDir + "/program/app/target/open-all-gpt.jar" +
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

    private void selectModel(PrintStream OUT, Arguments arguments) throws Exception
    {
        var modelsJson = readTextFile(arguments.getConfigRoot() + "/models.json");
        var typeRef = new TypeReference<Map<String, Models>>(){};
        var models = new ObjectMapper().readValue(modelsJson, typeRef);

        var modelSelector = new ModelSelector(OUT, models);
        var modelId = modelSelector.select();
        arguments.setModelId(modelId);
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
}
