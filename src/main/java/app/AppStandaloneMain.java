package app;

import config.Arguments;
import config.Config;
import config.ModelConfig;
import config.TokenizerConfig;
import parameters.safetensors.SafetensorsReader;
import tokenizer.TokenizerType;
import transformer.TransformerFlow;
import transformer.parallel.ParallelTransformerFlow;
import transformers.Talk;
import transformer.serial.SerialTransformerFlow;
import transformer.TransformerType;

import java.io.*;

import static parameters.FileUtil.checkFiles;
import static math.MathUtil.MATH;
import static ui.ConsoleUtil.getPrintStream;
import static ui.Logo.logo;
import static ui.TextUtil.toCenter;

public class AppStandaloneMain
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            logo(OUT,"Open All GPT", "CWgY-CWX-ygb", 'W');
            OUT.println(toCenter("Standalone app\n", 60));

            OUT.println(toCenter("Math module: " + MATH.getMathProviderName() + "\n", 60));

            new AppStandaloneMain().start(args);
        }
        catch (Throwable e)
        {
            OUT.println("ERROR: " + e.getMessage());
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

        OUT.println("\nLoading parameters... ");

        var tokenizer = TokenizerType.getTokenizer(tokenizerConfig);

        TransformerFlow flow = config.isParallel()
                ? new ParallelTransformerFlow(config, tokenizer)
                : new SerialTransformerFlow(config, tokenizer);

        OUT.println("... Parameters are loaded.");
        OUT.println("Parameter size:  " + Math.round((float) flow.getParameterSize() / 1000_000) + "M");
        OUT.println("Transformer flow: " + flow.getFlowType());

        if (!config.isCalculationOnly())
        {
            if (config.getQuantizationConfig() != null)
            {
                var quantizationType = config.getQuantizationConfig().getQuantizationType();
                if (quantizationType != null)
                {
                    OUT.print("\nThe model is quantized, using: " + quantizationType);

                    if (config.getQuantizationConfig().getDeQuantizeOnLoad())
                    {
                        OUT.println(", but it was de-quantized at load.");
                    }
                    else
                    {
                        OUT.println();
                    }
                }
            }
            else if (config.getQuantizeConfig() != null)
            {
                var quantizationType = config.getQuantizeConfig().getQuantizationType();
                if (quantizationType != null)
                {
                    OUT.println("\nIt was a non-quantized model, but we quantized it at load, using: " + quantizationType);
                }
            }

            Talk.talk(OUT, flow);
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
