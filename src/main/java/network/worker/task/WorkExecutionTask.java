package network.worker.task;

import network.Address;
import network.info.DecoderBlockType;
import network.info.input.HiddenStateInput;
import network.info.input.Input;
import network.info.input.TokenInput;
import network.info.output.EmptyOutput;
import network.info.output.HiddenStateOutput;
import network.info.output.Output;
import network.info.output.TokenOutput;
import network.message.toWorker.WorkMessage;
import network.message.toWorker.WorkResultMessage;
import math.dataType.vector.Vector;
import network.worker.state.WorkerState;

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

            var transformer = WorkerState.getWorkerState().getActiveModel(workMessage.getModelId());

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

            var hasOutput = true;
            if (segmentType.hasLayer())
            {
                for (var decoderBlock : workSegment.getDecoderBlocks())
                {
                    if (decoderBlock.getBlockType().equals(DecoderBlockType.ATTENTION_LAYER))
                    {
                        var attentionLayer = transformer.getAttentionLayer(decoderBlock.getDecoderId());
                        hiddenState = attentionLayer.process(hiddenState, isInputOnly);
                    }
                    else if (decoderBlock.getBlockType().equals(DecoderBlockType.NEURAL_NET_LAYER))
                    {
                        if (isInputOnly && decoderBlock.getLastDecoder())
                        {
                            hasOutput = false;
                        }
                        else
                        {
                            var neuralNetLayer = transformer.getNeuralNetLayer(decoderBlock.getDecoderId());
                            hiddenState = neuralNetLayer.process(hiddenState);
                        }
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
                {
                    output = new EmptyOutput();
                }
                else
                {
                    output = new TokenOutput(token);
                }
            }
            else
            {
                if (hasOutput)
                {
                    output = new HiddenStateOutput(hiddenState.getFloatType(), hiddenState.getValues());
                }
                else
                {
                    output = new EmptyOutput();
                }
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
