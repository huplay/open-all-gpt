package huplay.file;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class DataTypeUtil
{
    public static long toLittleEndian(long value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putLong(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getLong(0);
    }

    public static float toLittleEndian(float value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putFloat(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getFloat(0);
    }
}
