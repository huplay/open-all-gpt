package huplay.file.safetensors;

import java.util.List;

public class SafetensorsHeader
{
    private final String fileName;
    private final String id;
    private final long dataOffset;
    private final String format;
    private final SafetensorsDataType dataType;
    private final List<Integer> shape;
    private final long startOffset;
    private final long endOffset;

    public SafetensorsHeader(String fileName, String id, long dataOffset, String format, SafetensorsDataType dataType,
                             List<Integer> shape, long startOffset, long endOffset)
    {
        this.fileName = fileName;
        this.id = id;
        this.dataOffset = dataOffset;
        this.format = format;
        this.dataType = dataType;
        this.shape = shape;
        this.startOffset = startOffset;
        this.endOffset = endOffset;
    }

    public String getFileName()
    {
        return fileName;
    }

    // Getters
    public String getId() {return id;}
    public long getDataOffset() {return dataOffset;}
    public String getFormat() {return format;}
    public SafetensorsDataType getDataType() {return dataType;}
    public List<Integer> getShape() {return shape;}
    public long getStartOffset() {return startOffset;}
    public long getEndOffset() {return endOffset;}

    public long getSizeInBytes()
    {
        return endOffset - startOffset;
    }

    @Override
    public String toString()
    {
        return "ParameterDescriptor{" +
                "fileName='" + fileName + '\'' +
                ", id='" + id + '\'' +
                ", dataOffset=" + dataOffset +
                ", format='" + format + '\'' +
                ", dataType=" + dataType +
                ", shape=" + shape +
                ", startOffset=" + startOffset +
                ", endOffset=" + endOffset +
                '}';
    }
}
