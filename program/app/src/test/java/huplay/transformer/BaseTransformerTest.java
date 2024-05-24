package huplay.transformer;

import huplay.BaseTest;
import huplay.config.*;
import huplay.file.SafetensorsReader;

import java.io.File;

public class BaseTransformerTest extends BaseTest
{
    protected Config getTestConfig(String relativePath)
    {
        var resourcesDirectory = new File("src/test/resources");
        var root = resourcesDirectory.getAbsolutePath();

        var arguments = new Arguments(root, root, relativePath, 25, 40,
                false, 0, null, -1);

        var modelConfig = ModelConfig.read(arguments.getConfigPath(), arguments.getModelPath());
        var tokenizerConfig = TokenizerConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        var reader = new SafetensorsReader(arguments.getModelPath());
        return Config.readConfig(arguments, modelConfig, tokenizerConfig, reader);
    }
}
