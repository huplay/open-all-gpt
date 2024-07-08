package transformers;

import config.Arguments;
import config.Config;
import config.ModelConfig;
import config.TokenizerConfig;
import base.BaseTest;
import math.dataType.DataType;
import math.dataType.vector.Vector;
import parameters.safetensors.SafetensorsReader;
import transformer.TransformerType;
import transformer.serial.BaseAttentionLayer;
import transformer.serial.BaseNeuralNetLayer;
import transformer.serial.BaseTransformer;
import transformers._2019_02_openai_gpt2.serial.GPT2;

import java.io.File;

public class BaseTransformerTest extends BaseTest
{
    protected Config getTestConfig(String relativePath)
    {
        var resourcesDirectory = new File("src/test/resources");
        var root = resourcesDirectory.getAbsolutePath();

        var arguments = new Arguments(root, root, relativePath, 25, 40,
                false, 0, false, null, -1);

        var modelConfig = ModelConfig.read(arguments.getConfigPath(), arguments.getModelPath());
        var tokenizerConfig = TokenizerConfig.read(arguments.getConfigPath(), arguments.getModelPath());

        var reader = new SafetensorsReader(arguments.getModelPath());
        return Config.readConfig(arguments, modelConfig, tokenizerConfig, reader);
    }

    protected BaseTransformer getTestTransformer(String path)
    {
        var config = getTestConfig(path);
        var transformer = new GPT2();
        transformer.init(config);
        transformer.initDecoders();

        return transformer;
    }

    protected BaseAttentionLayer getTestAttentionLayer(String path)
    {
        var config = getTestConfig(path);
        var attentionLayer = TransformerType.getAttentionLayer(config.getTransformerType());
        attentionLayer.init(config, 0);

        return attentionLayer;
    }

    protected BaseNeuralNetLayer getTestNeuralNetLayer(String path)
    {
        var config = getTestConfig(path);
        var neuralNetLayer = TransformerType.getNeuralNetLayer(config.getTransformerType());
        neuralNetLayer.init(config, 0);

        return neuralNetLayer;
    }

    protected Vector getTestVector(float... values)
    {
        return Vector.of(DataType.FLOAT_32, values);
    }
}
