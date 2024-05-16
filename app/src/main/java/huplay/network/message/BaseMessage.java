package huplay.network.message;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import huplay.network.message.toServer.fromClient.*;
import huplay.network.message.toWorker.WorkRequest;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.network.message.toWorker.LoadModelRequest;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = Acknowledge.class, name = "Acknowledge"),
        @JsonSubTypes.Type(value = WorkerJoinedMessage.class, name = "WorkerJoinedMessage"),
        @JsonSubTypes.Type(value = PollOpenModel.class, name = "OpenModelRequest"),
        @JsonSubTypes.Type(value = PollOpenModelResponse.class, name = "OpenModelResponse"),
        @JsonSubTypes.Type(value = LoadModelRequest.class, name = "LoadModelRequest"),
        @JsonSubTypes.Type(value = ModelLoadedMessage.class, name = "ModelLoadedMessage"),
        @JsonSubTypes.Type(value = StartSessionRequest.class, name = "StartSessionRequest"),
        @JsonSubTypes.Type(value = StartSessionResponse.class, name = "StartSessionResponse"),
        @JsonSubTypes.Type(value = QueryRequest.class, name = "QueryRequest"),
        @JsonSubTypes.Type(value = QueryResponse.class, name = "QueryResponse"),
        @JsonSubTypes.Type(value = WorkRequest.class, name = "WorkRequest"),
        @JsonSubTypes.Type(value = WorkResultMessage.class, name = "WorkResultMessage"),
        @JsonSubTypes.Type(value = PollQueryResult.class, name = "GetQueryResultRequest"),
        @JsonSubTypes.Type(value = PollQueryResultResponse.class, name = "GetQueryResultResponse")
})
@JsonIgnoreProperties(ignoreUnknown = true)
public abstract class BaseMessage
{
}
