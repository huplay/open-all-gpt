package network.server.state;

import network.info.WorkSegment;
import tokenizer.Tokenizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelState
{
    private Tokenizer tokenizer;
    private List<WorkSegment> workSegments;

    // Key: UUID, value: text description
    private final Map<String, String> pendingTasks = new HashMap<>();

    public ModelState()
    {
    }

    // Getters
    public List<WorkSegment> getWorkSegments() {return workSegments;}
    public Tokenizer getTokenizer() {return tokenizer;}
    public Map<String, String> getPendingTasks() {return pendingTasks;}

    // Setters
    public void setTokenizer(Tokenizer tokenizer) {this.tokenizer = tokenizer;}
    public void setWorkSegments(List<WorkSegment> workSegments) {this.workSegments = workSegments;}
}
