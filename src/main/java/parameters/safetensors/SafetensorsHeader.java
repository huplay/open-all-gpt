package parameters.safetensors;

import java.util.List;

public class SafetensorsHeader
{
    private final String fileName;
    private final String parameterId;
    private final long dataOffset;
    private final String format;
    private final SafetensorsDataType dataType;
    private final List<Integer> shape;
    private final long startOffset;
    private final long endOffset;

    public SafetensorsHeader(String fileName, String parameterId, long dataOffset, String format, SafetensorsDataType dataType,
                             List<Integer> shape, long startOffset, long endOffset)
    {
        this.fileName = fileName;
        this.parameterId = parameterId;
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
    public String getParameterId() {return parameterId;}
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

    public long getOffset()
    {
        return dataOffset + startOffset;
    }

    @Override
    public String toString()
    {
        return "ParameterDescriptor{" +
                "fileName='" + fileName + '\'' +
                ", parameterId='" + parameterId + '\'' +
                ", dataOffset=" + dataOffset +
                ", format='" + format + '\'' +
                ", dataType=" + dataType +
                ", shape=" + shape +
                ", startOffset=" + startOffset +
                ", endOffset=" + endOffset +
                '}';
    }
}
