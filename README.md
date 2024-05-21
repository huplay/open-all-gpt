# Open All GPT

This is a demo application which implements different large language models in Java for learning purposes.

The main goal is to demonstrate the different decoder-only transformer architectures (without training), not to create an optimized application.

The second goal was to be able to execute as many models as possible. But it is limited by the operative memory, so I created a network framework which can split the task for multiple computers.

There's no parallel execution, so it is only a demonstration (and test) whether the used algorithm is correct for larger models.

The core transformer architecture is implemented multiple times for different models. The differences are mostly small, so it would be possible to make a single, more general implementation, but then handling the differences would occupy most of the code, so the separate implementations are more clear. Additionally, you can easily compare these, and at the beginning of the code I listed the differences in comments. 

The models use different data types (float32, float16, bfloat16). In Java there's only 32 (and 64) bit size float type, so currently all values are stored and calculated as 32 bit float values. But I wrapped these into a custom Vector class, so with tricks there's a chance to deal with the different data types. (Achieving less memory usage but more computation because of the conversions.) 

The core mathematical utility is implemented in three versions. There's a `standard` variant, which calculates everything using the simplest way. But I added an `ND4J` and a `Vector-API` implementation also, which can help a little bit to make the inference faster.

There's a standalone version, which has a text-only user interface (in console). But there's a network framework which consists of a server, one or more workers, and a client. The client has a text-only and a web based variant. So it is possible to try the inference from a browser, or even using a mobile phone.

<p align="center"><img src="static/screenshot.png" height="300"/></p>

## Install ##

1. Download and unzip this module: https://github.com/huplay/open-all-gpt

   (Or using git: ```git clone https://github.com/huplay/open-all-gpt.git```)

2. Install Java. (Version 21 or above)

## Run - Standalone mode ##

Using a command line tool (`cmd`) enter into the main directory:

   ```cd open-all-gpt```

   And start the app:

   ```run``` (On Windows)

This will open the `Launcher` app where the user can select the model. It opens the `Main` app which actually processes the user queries. (It was necessary to change dynamically the available memory.) 

## Run - Network mode ##

To use the network feature you have to start the `Server` app one or more `Worker` apps as well. These apps can run on different machines. (You can specify which ports the Server and Worker instances should listen. And on the Worker apps you have to tell the IP address of the Server.)

When the `Server` (`server.bat`) and `Worker` (`worker.bat`) nodes are running you can launch the `Client` (`client.bat`) which connects to the `Server`.

There's a `netrun.bat` to start a server, workers and the client. But for a real use you should use one worker per machine, configured to use as many heap memory as available.

You can use a browser instead of the text-based client, just target the server ip and port, using an url like this: 

   `http://serverhost:port/open-all-gpt`

## Trained parameters ##

The parameter files should be provided in `safetensors` file format, but the app can download these from the configured repository.

## Configuration ##

Every ported model has a subfolder within the `models` folder. (Organised in subfolders.)

The relative path (separated by the `/` character) will be the `modelId`, for example: `OpenAI/GPT-2/LARGE`.

Every model should have two files to configure the model itself and the tokenizer:

`model.json` format:
- `name`: The name of the model
- `date`: The release date of the model
- `transformerType`: The type of the transformer as it is listed in TransformerType.java
- `repo`: The url of the repo to download the model files (Currently it should be a Hugging Face repo)
- `branch`: The branch name of the repo
- `files`: list of necessary files. If some of these are missing it will be downloaded from the repo
- `parameterNaming`: The format of the transformer parameters (outside of decoders). The `{name}` will be replaced by the name used in the code
- `decoderParameterNaming`: The format of the transformer parameter at decoders. The `{decoderId}` will be replaced by the index of the decoder, the `{name}" by the name used in the code
- `memorySize`: The recommended minimum memory size to load the model
- `configOverride`: In the rare cases there's no `config.json` provided in the repo it is possible to give the parameters here


`tokenizer.json` format:
- `tokenizerType`: The type of the tokenizer as it is listed in TokenizerType.java
- `repo`: The url of the repo to download the model files (Currently it should be a Hugging Face repo)
- `branch`: The branch name of the repo
- `files`: list of necessary files. If some of these are missing it will be downloaded from the repo

The `tranformerType` and `tokenizerType` values are necessary, all other values are optional.

It is possible to provide a `folders.json` file in the parent folders tell the recommended displaying order or to make a model disabled.


## Supported Transformer types ##

The Transformer implementations can be found in the `transformer` package. Path: `app/src/main/java/huplay/transformer`.

Every transformer is implemented via three classes. The main is a subclass of the `BaseTransformer`, the attention block implementation is a subclass of `BaseAttentionLayer` and the neural net block is a `BaseNeuralNetLayer`.

There is no logic in the base classes, these are just helpers to store and access the configuration, the parameters and so on.

The following transformer architectures are implemented:

- `ORIGINAL_TRANSFORMER`: The first transformer, created by Google Brain in 2017. Described in the `Attention Is All You Need` paper. (The trained parameters are not published.)
- `OPENAI_GPT_1`: The first GPT created by OpenAI, released in June 2018. (Based on the Google's unpublished model, described in the `Attention Is All You Need` paper)
- `OPENAI_GPT_2`: The second GPT created by OpenAI, limited release in Feb 2019, full access in Nov 2019. Minor differences to GPT-1, only related to the normalization.
- `ELEUTHERAI_GPT_NEO`: First GPT implementation by EleutherAI, released in 2021. Almost the same as GPT-2 and GPT-3, few unimportant differences. 
- `ELEUTHERAI_GPT_J`: Second GPT implementation by EleutherAI, released in 2022. The biggest change to the previous architecture is the Rotational Position Embedding (RoPE).
- `BIG_SCIENCE_BLOOM`: Created by an open community organised by Hugging Face to create a similar model to GPT-3. Released in March-July 2022. The main difference to GPT-2/GPT-3 is the Alibi position embedding.
- `META_LLAMA`: Created by Meta (Facebook), released in Feb 2023. Currently only the original architecture is supported, but the latest models use Grouped Query Attention. Changes to GPT-2: Rotary position embedding, 3 layered MLP block, Swiglu activation function, RSM normalisation.

## Transformer implementation ##

The transformer architecture is implemented multiple times. All variant placed in a separate package within the `transformer` package as you can see highlighted in the image below:

<p align="center"><img src="static/transformer-packages.png" height="300"/></p>

(Every package name contains the release date and the company as well.) On the screenshot the GPT2 package was opened, but all others contain three Java classes:

- Main class (`GPT.java`)
- Attention layer class (First block of a decoder, `GPT2AttentionLayer.java`)
- Neural net layer class (Second block of a decoder, `GPT2NeuralNetLayer.java`)

The transformer architecture has a frame, and within that a series of decoders. The frame is implemented in the Main class, the decoder is in two classes, because every decoder has two blocks, an attention and a neural net (MLP) block.

Methods:

Main:
  - `loadParameters()`: Loads the main parameters into memory. This is called once, before the inference.
  - `preProcessToken()`: This is the beginning of the execution. Accepts a token and executes the pre process steps, like position embedding and normalization
  - //// The stack of decoders are called between the preProcessToken() and generateToken() ////
  - `generateToken()`: This is the end of the execution. Accepts the result of the decoder stack and generates a token.

Attention layer:
   - `loadParameters()`: Loads the parameters for this block into memory. This is called once, before the inference.
   - `process()`: The frame of the attention block. Contains normalization, residual connection and calls the attention implementation.
   - `attention()`: The implementation of the attention mechanism.

Neural net layer:
   - `loadParameters()`: Loads the parameters for this block into memory. This is called once, before the inference.
   - `process()`: The frame of the neural net (MLP) block. Contains normalization, residual connection and calls the core neural net implementation.
   - `neuralNet()`: The implementation of the core neural net layers.

If you take a look you can see all methods are very simple, only the attention mechanism is more complicated. Although, these uses some inherited methods from the parent (base) classes, and utility methods implementing the core mathematical algorithms, but those are also very simple codes, and everything is included in this repo.

<img src="static/java-main.png" height="300"/>
<img src="static/java-attention1.png" height="300"/>
<img src="static/java-attention2.png" height="300"/>
<img src="static/java-neural-net.png" height="300"/>

## For developers ##

Steps is you want to modify and rebuild the app:

1. Install Maven. (Java compile/build tool) (3.8.6 used during development).

2. Compile (build) the application. There are 3 possibilities, based on that which utility implementation you want to use.
   Standard: 

   ```mvn clean install```

   (This is equivalent to ```mvn clean install -Pstandard```)

   Using Nd4j:

   ```mvn clean install -Pnd4j```

   Using Java Vector API:

   ```mvn clean install -Pvector-api```


## Customization ##

At default the parameter files are downloaded to the `download` folder. You can specify a different folder using the ```OPEN_ALL_GPT_DOWNLOAD_ROOT``` environment variable.

Without using the menu, you can directly select the model, where the ```<model-name>``` is the full path within the model root:

   ```run <model-name>``` (On Windows)
    
Or on any systems:```java -jar target/open-all-gpt.jar <model-name>```

(To specify the order the folders has a (nn) prefix, which is removed when the folder name is displayed, but it is part of the model path.)

If you want to use the Vector API version (in the case you installed that variant) you have to use the ``runv <model-name>`` command.
This is necessary because the Vector API isn't ready (as of Java 20), added only as an incubator module, so we have to execute the Java Virtual Machine telling we want to use this incubator feature. 
  
## Additional command line parameters (Standalone version) ##

- `-max` - Maximum number of generated tokens (default: 25)
- `-topK` - Number of possibilities to chose from as next token (default: 40)
- `-calc` - Calculation only (without executing the model, it just displays the parameter size)

Example:

`run GPT2/XL -max=1024 -topk=100`

## Usage (Standalone version) ##

Actually there are two applications. The launcher app implements the model selection, which opens the main app in a separate window.

It is necessary to set the correct heap size (memory) for the main app, which depends on the selected model.

The main app shows a prompt, where you can provide a text:

```Input text:```

You can leave it empty, or type something, which will be continued by the system. While the input tokens are processed a `.` character is displayed. (One for every token.)
After that the system prints the generated tokens (one by one). If the maximum length is reached, or the response finished by an `END-OF-TEXT` token, a new prompt will be given.

Normally every prompt starts a completely new session (the state is cleared), but if you want to remain in the same context, start your input text by `+`.
If you use only a single `+` character, without more content, the system will continue the text as it would do without the limit of the max length.

To quit press Ctrl + C.

If the response contained special unicode characters, where a single character is constructed using multiple tokens, then the "one by one" printing solution will show "?" characters. But after the text is fully generated the whole text will be printed again to show the correct characters. (Only at cases when the original print wasn't perfect.) 


## Tokenizer ##

All LLMs use a byte pair encoding logic, but there are different versions.

Supported tokenizers: 
   - GPT-1
   - GPT-2 (used by GPT-3 as well), and for BLOOM with different vocabulary
   - SentencePiece (used by Llama1 and Llama2)
