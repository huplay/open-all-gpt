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
import huplay.network.message.toWorker.WorkRequest;
import huplay.network.message.toWorker.WorkResultMessage;
import huplay.util.FloatType;
import huplay.util.Vector;

import static huplay.network.worker.state.WorkerState.getWorkerState;

public class WorkExecutionTask implements Runnable
{
    private final WorkRequest workRequest;
    private final Address server;

    public WorkExecutionTask(WorkRequest workRequest, Address server)
    {
        this.workRequest = workRequest;
        this.server = server;
    }

    @Override
    public void run()
    {
        try
        {
            Input input = workRequest.getInput();

            var workSegment = workRequest.getWorkSegment();
            var segmentType = workSegment.getWorkSegmentType();

            // TODO: Validate the input to the input of the segment Type

            boolean isInputOnly = workRequest.getInputOnly();

            var transformer = getWorkerState().getActiveModel(workRequest.getModelId());

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
                hiddenState = new Vector(FloatType.FLOAT32, hiddenStateInput.getHiddenState());
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
                        var attentionLayer = transformer.getAttentionLayers().get(decoderBlock.getDecoderId());
                        hiddenState = attentionLayer.process(hiddenState, isInputOnly);
                    }
                    else if (decoderBlock.getBlockType().equals(DecoderBlockType.NEURAL_NET_LAYER))
                    {
                        var neuralNetLayer = transformer.getNeuralNetLayers().get(decoderBlock.getDecoderId());
                        hiddenState = neuralNetLayer.process(hiddenState, isInputOnly);
                    }
                }
            }

            if (!isInputOnly && segmentType.hasTail())
            {
                token = transformer.generateToken(hiddenState, workRequest.getTopK());
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
                    output = new HiddenStateOutput(hiddenState.getFloat32Values());

            }

            WorkResultMessage result = new WorkResultMessage(workRequest.getWorkUUID(), workSegment, output);
            result.send(server);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e); // TODO: Handle errors
        }
    }
}
