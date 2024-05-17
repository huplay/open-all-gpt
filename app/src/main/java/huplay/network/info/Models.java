package huplay.network.info;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Models
{
    private int order;
    private String date;
    private String size;
    private boolean disabled;
    private Map<String, Models> folders;

    // Getter
    public int getOrder() {return order;}
    public String getDate() {return date;}
    public String getSize() {return size;}
    public boolean getDisabled() {return disabled;}
    public Map<String, Models> getFolders() {return folders;}

    @Override
    public String toString()
    {
        return "ModelFolderDetails{" +
                "order=" + order +
                ", date='" + date + '\'' +
                ", size='" + size + '\'' +
                ", disabled=" + disabled +
                ", folders=" + folders +
                '}';
    }
}
