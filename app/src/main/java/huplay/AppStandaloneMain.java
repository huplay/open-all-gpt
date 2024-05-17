package huplay;

import huplay.config.*;
import huplay.file.SafetensorsReader;
import huplay.tokenizer.TokenizerType;
import huplay.transformer.Talk;
import huplay.transformer.TransformerFlow;
import huplay.transformer.TransformerType;

import java.io.*;
import java.util.Arrays;

import static huplay.AppNetworkClient.UTIL;
import static huplay.file.FileUtil.checkFiles;
import static huplay.ui.ConsoleUtil.getPrintStream;
import static huplay.ui.Logo.logo;
import static huplay.ui.TextUtil.toCenter;

public class AppStandaloneMain
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            logo(OUT,"Open All GPT", "CWgY-CWX-ygb", 'W');
            OUT.println(toCenter("Standalone app\n", 60));

            OUT.println(toCenter("Util: " + UTIL.getUtilName() + "\n", 60));

            new AppStandaloneMain().start(args);
        }
        catch (IdentifiedException e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
        catch (Throwable e)
        {
            var stackTraceElements = e.getStackTrace();
            if (stackTraceElements != null)
            {
                for (var element : stackTraceElements)
                {
                    OUT.println(element.toString());
                }
            }
            OUT.println("ERROR: " + e.getMessage() + " " + Arrays.toString(e.getStackTrace()));
        }
    }

    private void start(String... args)
    {
        // Read arguments
        var arguments = Arguments.read(args);

        // Read the modelConfig of the selected model
        var modelConfig = ModelConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        // Check necessary files
        var missingFiles = checkFiles(modelConfig, arguments.getModelPath());
        if (!missingFiles.isEmpty())
        {
            throw new IdentifiedException("There are missing files: " + missingFiles);
        }

        var tokenizerConfig = TokenizerConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        missingFiles = checkFiles(tokenizerConfig, arguments.getModelPath());
        if (!missingFiles.isEmpty())
        {
            throw new IdentifiedException("There are missing files: " + missingFiles);
        }

        // Create the parameter reader
        var reader = new SafetensorsReader(arguments.getModelPath());

        // Read the config (first look into the model folder, second to the config folder (maybe it's different)
        var config = Config.readConfig(arguments, modelConfig, tokenizerConfig, reader);

        displayConfig(config, 0);

        OUT.print("\nLoading parameters... ");
        var transformer = TransformerType.getTransformer(config.getTransformerType());

        var tokenizer = TokenizerType.getTokenizer(tokenizerConfig);
        var transformerFlow = new TransformerFlow(config, tokenizer, transformer);

        OUT.println("Done.");
        OUT.println("Parameter size:  " + Math.round((float) transformer.getParameterSize() / 1000_000) + "M");

        if (!config.isCalculationOnly())
        {
            Talk.talk(OUT, transformerFlow);
        }
    }

    public static void displayConfig(Config config, long parameterSize)
    {
        // Print settings
        OUT.println("Model: " + config.getName());
        OUT.println("Path: " + config.getModelPath());
        if (parameterSize > 0)
        {
            OUT.print("Number of parameters: " + Math.round(parameterSize / 1000_000d) + "M ");
        }
        OUT.println("Hidden size: " + config.getHiddenSize() +
                ", decoders: " + config.getDecoderCount() +
                ", heads: " + config.getHeadCount() +
                ", head size: " + config.getHeadSize());

        OUT.println("Maximum length of generated text: " + config.getLengthLimit());
        OUT.println("Output is selected from the best " + config.getTopK() + " tokens (topK)");
        OUT.println("Max memory: " + config.getMemorySize());
    }
}
