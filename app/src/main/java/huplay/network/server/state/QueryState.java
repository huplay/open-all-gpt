package huplay.network.server.state;

import huplay.network.info.WorkSegment;
import huplay.tokenizer.Token;

import java.util.ArrayList;
import java.util.List;

public class QueryState
{
    private final String modelId;
    private final String sessionUUID;
    private final String queryUUID;
    private final int topK;
    private final int maxLength;
    private final List<Token> inputTokens;

    // Starting values for a query processing:
    private int processedInputCount = 0;
    private final List<Token> generatedTokens = new ArrayList<>();
    private String generatedText = "";

    private final List<WorkSegment> workSegments;
    private int workSegmentIndex = 0;

    public QueryState(String modelId, String sessionUUID, String queryUUID, List<Token> inputTokens,
                      int topK, int maxLength, List<WorkSegment> workSegments)
    {
        this.modelId = modelId;
        this.sessionUUID = sessionUUID;
        this.queryUUID = queryUUID;
        this.topK = topK;
        this.maxLength = maxLength;
        this.inputTokens = inputTokens;
        this.workSegments = workSegments;
    }

    public void incrementProcessedInput()
    {
        processedInputCount++;
    }

    public int getActualInputToken()
    {
        return inputTokens.get(processedInputCount).getId();
    }

    public boolean isInputOnly()
    {
        return inputTokens.size() - 1 > processedInputCount;
    }

    public int getPos()
    {
        return processedInputCount + generatedTokens.size();
    }

    public boolean isMaxLengthReached()
    {
        return getGeneratedTokens().size() >= getMaxLength();
    }

    public WorkSegment getActualWorkSegment()
    {
        return workSegments.get(workSegmentIndex);
    }

    public WorkSegment nextWorkSegment()
    {
        workSegmentIndex++;
        return getActualWorkSegment();
    }

    public void resetWorkSegmentIndex()
    {
        workSegmentIndex = 0;
    }

    // Getters
    public String getModelId() {return modelId;}
    public String getSessionUUID() {return sessionUUID;}
    public String getQueryUUID() {return queryUUID;}
    public int getTopK() {return topK;}
    public int getMaxLength() {return maxLength;}
    public List<Token> getInputTokens() {return inputTokens;}
    public int getProcessedInputCount() {return processedInputCount;}
    public List<Token> getGeneratedTokens() {return generatedTokens;}
    public List<WorkSegment> getWorkSegments() {return workSegments;}
    public int getWorkSegmentIndex() {return workSegmentIndex;}
    public String getGeneratedText() {return generatedText;}
    public void setGeneratedText(String generatedText) {this.generatedText = generatedText;}

    public List<Integer> getAllTokens()
    {
        List<Integer> tokens = new ArrayList<>();

        for (Token token : inputTokens)
        {
            tokens.add(token.getId());
        }

        for (Token token : generatedTokens)
        {
            tokens.add(token.getId());
        }

        return tokens;
    }
}
