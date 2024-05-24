package huplay.network.message;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import huplay.network.message.toServer.fromClient.*;
import huplay.network.message.toWorker.WorkMessage;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.network.message.toWorker.LoadModelMessage;
import huplay.network.message.toServer.fromWorker.ModelLoadedMessage;
import huplay.network.message.toServer.fromWorker.WorkerJoinedMessage;

// For messages ending with "Message" the response contains nothing, so an Acknowledge sent only
// For other messages there's a "Request" and a "Response" class as well

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = Acknowledge.class, name = "Acknowledge"),
        @JsonSubTypes.Type(value = ClientJoinedRequest.class, name = "ClientJoinedRequest"),
        @JsonSubTypes.Type(value = ClientJoinedResponse.class, name = "ClientJoinedResponse"),
        @JsonSubTypes.Type(value = WorkerJoinedMessage.class, name = "WorkerJoinedMessage"),
        @JsonSubTypes.Type(value = PollOpenModelRequest.class, name = "PollOpenModelRequest"),
        @JsonSubTypes.Type(value = PollOpenModelResponse.class, name = "PollOpenModelResponse"),
        @JsonSubTypes.Type(value = LoadModelMessage.class, name = "LoadModelMessage"),
        @JsonSubTypes.Type(value = ModelLoadedMessage.class, name = "ModelLoadedMessage"),
        @JsonSubTypes.Type(value = StartSessionRequest.class, name = "StartSessionRequest"),
        @JsonSubTypes.Type(value = StartSessionResponse.class, name = "StartSessionResponse"),
        @JsonSubTypes.Type(value = QueryRequest.class, name = "QueryRequest"),
        @JsonSubTypes.Type(value = QueryResponse.class, name = "QueryResponse"),
        @JsonSubTypes.Type(value = WorkMessage.class, name = "WorkMessage"),
        @JsonSubTypes.Type(value = WorkResultMessage.class, name = "WorkResultMessage"),
        @JsonSubTypes.Type(value = PollQueryResultRequest.class, name = "PollQueryResultRequest"),
        @JsonSubTypes.Type(value = PollQueryResultResponse.class, name = "PollQueryResultResponse")
})
@JsonIgnoreProperties(ignoreUnknown = true)
public abstract class BaseMessage
{
}
