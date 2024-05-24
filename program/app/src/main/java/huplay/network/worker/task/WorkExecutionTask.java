package huplay.network.worker.task;

import huplay.network.Address;
import huplay.network.info.DecoderBlockType;
import huplay.network.info.input.HiddenStateInput;
import huplay.network.info.input.Input;
import huplay.network.info.input.TokenInput;
import huplay.network.info.output.EmptyOutput;
import huplay.network.info.output.HiddenStateOutput;
import huplay.network.info.output.Output;
import huplay.network.info.output.TokenOutput;
import huplay.network.message.toWorker.WorkMessage;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.dataType.vector.Vector;

import static huplay.network.worker.state.WorkerState.getWorkerState;

public class WorkExecutionTask implements Runnable
{
    private final WorkMessage workMessage;
    private final Address server;

    public WorkExecutionTask(WorkMessage workMessage, Address server)
    {
        this.workMessage = workMessage;
        this.server = server;
    }

    @Override
    public void run()
    {
        try
        {
            Input input = workMessage.getInput();

            var workSegment = workMessage.getWorkSegment();
            var segmentType = workSegment.getWorkSegmentType();

            // TODO: Validate the input to the input of the segment Type

            boolean isInputOnly = workMessage.getInputOnly();

            var transformer = getWorkerState().getActiveModel(workMessage.getModelId());

            Integer token = null;
            Integer pos = null;
            Vector hiddenState = null;

            if (input instanceof TokenInput tokenInput)
            {
                token = tokenInput.getToken();
                pos = tokenInput.getPos();
            }
            else if (input instanceof HiddenStateInput hiddenStateInput)
            {
                hiddenState = Vector.of(hiddenStateInput.getFloatType(), hiddenStateInput.getHiddenState());
            }

            if (segmentType.hasHead())
            {
                hiddenState = transformer.preProcessToken(pos, token);
            }

            if (segmentType.hasLayer())
            {
                for (var decoderBlock : workSegment.getDecoderBlocks())
                {
                    if (decoderBlock.getBlockType().equals(DecoderBlockType.ATTENTION_LAYER))
                    {
                        var attentionLayer = transformer.getAttentionLayers(decoderBlock.getDecoderId());
                        hiddenState = attentionLayer.process(hiddenState, isInputOnly);
                    }
                    else if (decoderBlock.getBlockType().equals(DecoderBlockType.NEURAL_NET_LAYER))
                    {
                        var neuralNetLayer = transformer.getNeuralNetLayers(decoderBlock.getDecoderId());
                        hiddenState = neuralNetLayer.process(hiddenState, isInputOnly);
                    }
                }
            }

            if (!isInputOnly && segmentType.hasTail())
            {
                token = transformer.generateToken(hiddenState, workMessage.getTopK());
            }

            // Determine which value should be the output
            Output output;
            if (segmentType.hasTail())
            {
                if (isInputOnly)
                    output = new EmptyOutput();
                else
                    output = new TokenOutput(token);
            }
            else
            {
                if (hiddenState == null)
                    output = new EmptyOutput();
                else
                    output = new HiddenStateOutput(hiddenState.getFloatType(), hiddenState.getValues());

            }

            WorkResultMessage result = new WorkResultMessage(workMessage.getWorkUUID(), workSegment, output);
            result.send(server);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e); // TODO: Handle errors
        }
    }
}
