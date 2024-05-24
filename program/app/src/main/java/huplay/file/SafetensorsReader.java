package huplay.file;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.dataType.matrix.Matrix;
import huplay.dataType.vector.BrainFloat16Vector;
import huplay.dataType.vector.Float16Vector;
import huplay.dataType.vector.Float32Vector;
import huplay.dataType.vector.Vector;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static huplay.file.FileUtil.checkHeaderFiles;
import static huplay.file.FileUtil.readTextFile;
import static huplay.transformer.TransformerUtil.UTIL;

/**
 * Reader of the trained parameters

 * The reader caches the header files in the header folder.
 * (This is the same content that can be found at the beginning of a safetensors file (after the 8 size bytes.))
 * This helps the Server to determine the model sizes (using the reader in calculateOnly mode),
 * and this way it needs only the header file, without downloading the whole safetensors file.
 */
public class SafetensorsReader
{
    private final Map<String, SafetensorsHeader> parameterHeaders = new HashMap<>();

    public SafetensorsReader(String downloadPath)
    {
        // Check the model folder
        var downloadFolder = new File(downloadPath);
        if (!downloadFolder.exists() || !downloadFolder.isDirectory())
        {
            throw new IdentifiedException("Download folder not found: " + downloadPath);
        }

        // Check weather the header file is created for all safetensors file...
        if (!checkHeaderFiles(downloadPath, downloadPath + "/header"))
        {
            // ... if not, create it
            createHeadersFromSafetensorsFiles(downloadFolder);
        }

        // Read the header files into memory
        for (var file : downloadFolder.listFiles())
        {
            if (file.isFile() && file.getName().endsWith("safetensors"))
            {
                String headerFilePath = downloadFolder.getAbsolutePath() + "/header/" + file.getName() + ".header";
                var header = readTextFile(headerFilePath);
                deserializeHeader(file.getAbsolutePath(), header);
            }
        }
    }

    private void createHeadersFromSafetensorsFiles(File modelFolder)
    {
        // Create header files for all safetensors files
        for (var file : modelFolder.listFiles())
        {
            if (file.isFile() && file.getName().endsWith("safetensors"))
            {
                var headerString = readSafetensorsHeader(file.getAbsolutePath());

                writeHeaderFile(modelFolder + "/header", file.getName() + ".header", headerString);
                // TODO save the headerString as safetensors.header into the header folder. Create it if missing
            }
        }
    }

    private void writeHeaderFile(String folderPath, String fileName, String content)
    {
        var filePath = folderPath + "/" + fileName;
        try
        {
            // Create header folder if missing
            var path = Paths.get(folderPath);
            Files.createDirectories(path);

            FileWriter fileWriter = new FileWriter(filePath);
            fileWriter.write(content);
            fileWriter.close();
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot create header file: " + filePath);
        }
    }

    public String readSafetensorsHeader(String fileName)
    {
        var headerSize = readHeaderSize(fileName);

        var array = new byte[(int)headerSize];

        try (var stream = new FileInputStream(fileName))
        {
            var channel = stream.getChannel();
            var buffer = channel.map(FileChannel.MapMode.READ_ONLY, 8, headerSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            ByteBuffer byteBuffer = buffer.asReadOnlyBuffer();

            byteBuffer.get(array, 0, (int)headerSize);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Parameter file read error reading safetensors file header. (" + fileName + ")", e);
        }

        return new String(array, StandardCharsets.UTF_8);
    }

    private long readHeaderSize(String fileName)
    {
        var array = new long[1];

        try (var stream = new FileInputStream(fileName))
        {
            var channel = stream.getChannel();
            var buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, 8);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            LongBuffer longBuffer = buffer.asLongBuffer();

            longBuffer.get(array, 0, 1);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Parameter file read error reading header size. (" + fileName + ")", e);
        }

        return array[0];
    }

    public void deserializeHeader(String fileName, String headerString)
    {
        var tensors = new HashMap<String, SafetensorsModel.TensorModel>();

        try
        {
            var typeRef = new TypeReference<Map<String, SafetensorsModel.TensorModel>>(){};
            tensors.putAll(new ObjectMapper().readValue(headerString, typeRef));
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Parameter file read error during deserialization. (" + fileName + ")", e);
        }

        for (var entry : tensors.entrySet())
        {
            var id = entry.getKey();

            if (id.equals("__metadata__")) continue;

            var tensor = entry.getValue();

            var dataType = SafetensorsDataType.valueOf(tensor.getDataType());
            var shape = tensor.getShape();
            var offsets = tensor.getDataOffsets();

            if (offsets == null || offsets.size() != 2)
            {
                throw new IdentifiedException("Parameter file read error during reading offsets. (" + id + ")");
            }

            var dataOffset = headerString.length() + 8;
            var start = offsets.get(0);
            var end = offsets.get(1);
            var header = new SafetensorsHeader(fileName, id, dataOffset, "pt", dataType, shape, start, end);

            parameterHeaders.put(id, header);
        }
    }

    public Vector readVector(String file, int size)
    {
        return read(file, size);
    }

    public Matrix readMatrix(String file, int rows, int cols)
    {
        // TODO: Read quantized matrix

        var vector = read(file, rows * cols);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    public SafetensorsDataType getDataType(String file)
    {
        return parameterHeaders.get(file).getDataType();
    }

    private void checkSize(SafetensorsHeader header, long expectedSize)
    {
        var parameterSize = header.getSizeInBytes() * 8 / header.getDataType().getBits();
        if (parameterSize != expectedSize)
        {
            System.out.println("\nWARNING: The file has different size (" + parameterSize + ") " +
                    "to the expected (" + expectedSize + "). Id: " + header.getId());
        }
    }

    private Vector read(String id, int size)
    {
        var header = parameterHeaders.get(id);
        if (header == null)
        {
            throw new IdentifiedException("Header not found for key: " + id);
        }

        checkSize(header, size);

        var offset = header.getDataOffset() + header.getStartOffset();
        var file = new File(header.getFileName());

        try (var stream = new FileInputStream(file))
        {
            return switch (header.getDataType())
            {
                case F16    -> readFloat16(stream, size, offset);
                case BF16   -> readBrainFloat16(stream, size, offset);
                case F32    -> readFloat32(stream, size, offset);
                default ->
                        throw new IdentifiedException("Not supported data type: " + header.getDataType() + ", key: " + id);
            };
        }
        catch (IOException e)
        {
            throw new RuntimeException("Parameter file read error in " + header.getFileName() + ", key: " + id);
        }
    }

    private Vector readFloat32(FileInputStream stream, int size, long offset) throws IOException
    {
        var array = new float[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().get(array, 0, size);

        return new Float32Vector(array);
    }

    private Vector readFloat16(FileInputStream stream, int size, long offset) throws IOException
    {
        var array = new short[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asShortBuffer().get(array, 0, size);

        return new Float16Vector(array);
    }

    private Vector readBrainFloat16(FileInputStream stream, int size, long offset) throws IOException
    {
        var array = new short[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asShortBuffer().get(array, 0, size);

        return new BrainFloat16Vector(array);
    }
}
