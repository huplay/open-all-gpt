package parameters.safetensors;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import app.IdentifiedException;
import math.dataType.matrix.Matrix;
import math.dataType.matrix.VectorArrayMatrix;
import math.dataType.DataType;
import math.dataType.vector.BrainFloat16Vector;
import math.dataType.vector.Float16Vector;
import math.dataType.vector.Float32Vector;
import math.dataType.vector.Vector;
import parameters.ParameterReader;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static math.TypeConversionUtility.bytesToBoolean;
import static parameters.FileUtil.checkHeaderFiles;
import static parameters.FileUtil.readTextFile;

/**
 * Reader of the trained parameters

 * The reader caches the header files in the header folder.
 * (This is the same content that can be found at the beginning of a safetensors file (after the 8 size bytes.))
 * This helps the Server to determine the model sizes (using the reader in calculateOnly mode),
 * and this way it needs only the header file, without downloading the whole safetensors file.
 */
public class SafetensorsReader implements ParameterReader
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

    // Getter
    public Map<String, SafetensorsHeader> getParameterHeaders() {return parameterHeaders;}

    private void createHeadersFromSafetensorsFiles(File modelFolder)
    {
        // Create header files for all safetensors files
        for (var file : modelFolder.listFiles())
        {
            if (file.isFile() && file.getName().endsWith("safetensors"))
            {
                var headerString = readSafetensorsHeader(file.getAbsolutePath());

                writeHeaderFile(modelFolder + "/header", file.getName() + ".header", headerString);
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

        try (var stream = new FileInputStream(fileName))
        {
            return new String(readChannelAsByte(stream, 8, (int)headerSize), StandardCharsets.UTF_8);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Parameter file read error reading safetensors file header. (" + fileName + ")", e);
        }
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

    @Override
    public long getBits(String parameterId)
    {
       return getDataType(parameterId).getBits();
    }

    public DataType getDataType(String parameterId)
    {
        return parameterHeaders.get(parameterId).getDataType().getDataType();
    }

    private void checkSize(SafetensorsHeader header, long[] expectedShape)
    {
        var expectedSize = 1L;
        for (var dim : expectedShape)
        {
            expectedSize *= dim;
        }

        var actualSize = header.getSizeInBytes() * 8 / header.getDataType().getBits();

        if (expectedSize != actualSize)
        {
            System.out.println("WARNING: The parameter tensor has different size to the expected.\n" +
                    "         Id: " + header.getParameterId() + "\n" +
                    "         Expected size: " + expectedSize + ", shape: [" + shapeToString(expectedShape) + "]\n" +
                    "         Actual size:   " + actualSize + ", shape: [" + shapeToString(header.getShape()) + "]");
        }
        else
        {
            if (expectedShape.length != header.getShape().size())
            {
                System.out.println("WARNING: The parameter tensor has the same size, but different shape to the expected.\n" +
                        "         Id: " + header.getParameterId() + "\n" +
                        "         Expected shape: [" + shapeToString(expectedShape) + "]\n" +
                        "         Actual shape: [" + shapeToString(header.getShape()) + "]\n");
            }
        }
    }

    private String shapeToString(long[] shape)
    {
        return Arrays.stream(shape)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.joining(", "));
    }

    private String shapeToString(List<Integer> shape)
    {
        return shape.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(", "));
    }

    private SafetensorsHeader getHeader(String parameterId, long[] shape)
    {
        var header = parameterHeaders.get(parameterId);
        if (header == null)
        {
            throw new IdentifiedException("Header not found for key: " + parameterId);
        }

        checkSize(header, shape);

        return header;
    }

    public String readString(String parameterId)
    {
        var header = parameterHeaders.get(parameterId);
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var size = header.getShape().getFirst();
            return new String(readChannelAsByte(stream, header.getOffset(), size), StandardCharsets.UTF_8);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Error reading String parameter " + parameterId, e);
        }
    }

    @Override
    public int[] readIntArray(String parameterId, int size)
    {
        var header = getHeader(parameterId, new long[] {size});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            return readChannelAsInt(stream, header.getOffset(), size);
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading int array parameter " + parameterId, e);
        }
    }

    @Override
    public byte[] readByteArray(String parameterId, int size)
    {
        var header = getHeader(parameterId, new long[] {size});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            return readChannelAsByte(stream, header.getOffset(), size);
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading byte array parameter " + parameterId, e);
        }
    }

    @Override
    public float[] readFloatArray(String parameterId, int size)
    {
        var header = getHeader(parameterId, new long[] {size});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            return readChannelAsFloat(stream, header.getOffset(), size);
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading float array parameter " + parameterId, e);
        }
    }

    public short[] readShortArray(String parameterId, int size)
    {
        var header = getHeader(parameterId, new long[] {size});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            return readChannelAsShort(stream, header.getOffset(), size);
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading short (or float16) array parameter " + parameterId, e);
        }
    }

    public boolean[] readBooleanArray(String parameterId, int size)
    {
        var header = getHeader(parameterId, new long[] {size});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            return readChannelAsBoolean(stream, header.getOffset(), size);
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading short (or float16) array parameter " + parameterId, e);
        }
    }

    @Override
    public Vector readFloat32Vector(String parameterId, int size)
    {
        return new Float32Vector(readFloatArray(parameterId, size));
    }

    @Override
    public Vector readFloat16Vector(String parameterId, int size)
    {
        return new Float16Vector(readShortArray(parameterId, size));
    }

    @Override
    public Vector readBrainFloat16Vector(String parameterId, int size)
    {
        return new BrainFloat16Vector(readShortArray(parameterId, size));
    }

    @Override
    public int[][] readIntArray2D(String parameterId, int rows, int cols)
    {
        var matrix = new int[rows][cols];

        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                matrix[i] = readChannelAsInt(stream, offset, cols);

                rowOffset += cols * 4;
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading int 2D array parameter " + parameterId, e);
        }

        return matrix;
    }

    @Override
    public byte[][] readByteArray2D(String parameterId, int rows, int cols)
    {
        var matrix = new byte[rows][cols];

        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                var array = readChannelAsByte(stream, offset, cols);
                matrix[i] = array;

                rowOffset += cols;
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading byte 2D array parameter " + parameterId + e.getMessage(), e);
        }

        return matrix;
    }

    @Override
    public float[][] readFloatArray2D(String parameterId, int rows, int cols)
    {
        var matrix = new float[rows][cols];

        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                matrix[i] = readChannelAsFloat(stream, offset, cols);

                rowOffset += cols * 4;
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading float 2D array parameter " + parameterId, e);
        }

        return matrix;
    }

    @Override
    public short[][] readShortArray2D(String parameterId, int rows, int cols)
    {
        var matrix = new short[rows][cols];

        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                matrix[i] = readChannelAsShort(stream, offset, cols);

                rowOffset += cols * 4;
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading short 2D array parameter " + parameterId, e);
        }

        return matrix;
    }

    @Override
    public boolean[][] readBooleanArray2D(String parameterId, int rows, int cols)
    {
        var result = new boolean[1][rows * cols];

        result[0] = readBooleanArray(parameterId, rows * cols);

        return result;
    }

    @Override
    public Matrix readFloat32Matrix(String parameterId, int rows, int cols)
    {
        var matrix = new VectorArrayMatrix(DataType.FLOAT_32, rows, cols);

        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                var array = readChannelAsFloat(stream, offset, cols);
                matrix.setRow(i, new Float32Vector(array));

                rowOffset += cols * 4;
            }
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading float 32 matrix parameter " + parameterId, e);
        }

        return matrix;
    }

    @Override
    public Matrix readFloat16Matrix(String parameterId, int rows, int cols)
    {
        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var matrix = new VectorArrayMatrix(DataType.FLOAT_16, rows, cols);

            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                var array = readChannelAsShort(stream, offset, cols);
                matrix.setRow(i, new Float16Vector(array));

                rowOffset += cols * 2;
            }

            return matrix;
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading float 16 matrix parameter " + parameterId, e);
        }
    }

    @Override
    public Matrix readBrainFloat16Matrix(String parameterId, int rows, int cols)
    {
        var header = getHeader(parameterId, new long[] {rows, cols});
        try (var stream = new FileInputStream(header.getFileName()))
        {
            var matrix = new VectorArrayMatrix(DataType.BRAIN_FLOAT_16, rows, cols);

            var rowOffset = 0;
            for (int i = 0; i < rows; i++)
            {
                var offset = header.getOffset() + rowOffset;
                var array = readChannelAsShort(stream, offset, cols);
                matrix.setRow(i, new BrainFloat16Vector(array));

                rowOffset += cols * 2;
            }

            return matrix;
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Error reading brain float 16 matrix parameter " + parameterId, e);
        }
    }

    private byte[] readChannelAsByte(FileInputStream stream, long position, int size) throws IOException
    {
        var array = new byte[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, position, size);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.get(array, 0, size);

        return array;
    }

    private int[] readChannelAsInt(FileInputStream stream, long position, int size) throws IOException
    {
        var array = new int[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, position, (long)size * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asIntBuffer().get(array, 0, size);

        return array;
    }

    private float[] readChannelAsFloat(FileInputStream stream, long position, int size) throws IOException
    {
        var array = new float[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, position, (long)size * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().get(array, 0, size);

        return array;
    }

    private short[] readChannelAsShort(FileInputStream stream, long position, int size) throws IOException
    {
        var array = new short[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, position, (long)size * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asShortBuffer().get(array, 0, size);

        return array;
    }

    private boolean[] readChannelAsBoolean(FileInputStream stream, long position, int size) throws IOException
    {
        var array = new byte[size];

        var buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, position, size);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.get(array, 0, size);

        return bytesToBoolean(array);
    }
}
