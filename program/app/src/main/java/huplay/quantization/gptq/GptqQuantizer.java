package huplay.quantization.gptq;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.IdentifiedException;
import huplay.config.Config;
import huplay.config.ParameterType;
import huplay.dataType.matrix.Matrix;
import huplay.parameters.ParameterReader;
import huplay.quantization.AbstractQuantizer;
import huplay.quantization.QuantizedMatrix;

import static huplay.MathUtilProvider.MATH;
import static huplay.dataType.matrix.Matrix.emptyMatrix;
import static huplay.parameters.FileUtil.readTextFile;
import static huplay.math.TypeConversionUtility.*;

/**
 * Parameter reader for GPTQ quantization (GPTQ de-quantization)

 * GPTQ (GPT Post-Training Quantization) (ETH Zurich, IST Austria, NeuralMagic)
 * Publication (31 Oct 2022): https://arxiv.org/abs/2210.17323
 * Code for the publication: https://github.com/IST-DASLab/gptq

 * @author Hunor Szegi
 */
public class GptqQuantizer extends AbstractQuantizer
{
    private static final String GROUP_INDEX_KEY = "groupIndex";
    private static final String ZEROS_KEY = "zeros";
    private static final String SCALES_KEY = "scales";
    private static final String WEIGHTS_KEY = "weights";

    public GptqQuantizer(Config config)
    {
        super(config);

        addDefaultNaming(GROUP_INDEX_KEY, "{name-1}.g_idx");
        addDefaultNaming(ZEROS_KEY, "{name-1}.qzeros");
        addDefaultNaming(SCALES_KEY, "{name-1}.scales");
        addDefaultNaming(WEIGHTS_KEY, "{name-1}.qweight");
    }

    @Override
    public QuantizedMatrix quantize(ParameterType parameterType, Matrix matrix)
    {
        // TODO
        return null;
    }

    @Override
    public Matrix load(ParameterReader reader, ParameterType parameterType, String parameterId, int rows, int cols)
    {
        try
        {
            // Read the quantize_config.json
            var quantizeConfigString = readTextFile(config.getModelPath() + "/quantize_config.json");
            var gptqConfig = new ObjectMapper().readValue(quantizeConfigString, GptqConfig.class);

            var realRows = rows;
            var realCols = cols;

            if (parameterType.isVertical())
            {
                realRows = cols;
                realCols = rows;
            }

            var matrix = readMatrixInternal(gptqConfig, reader, parameterId, realRows, realCols);

            if (parameterType.isVertical())
            {
                return MATH.transposeMatrix(matrix);
            }
            else
            {
                return matrix;
            }
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Error during reading quantize_config.json for GPTQ", e);
        }
    }

    private Matrix readMatrixInternal(GptqConfig gptqConfig, ParameterReader reader, String id, int rows, int cols)
    {
        // Good blog: https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html

        /* "The resulting quantized weights are stored as a 2D numpy array called qweight,
            where the rows correspond to different quantization chunks and the columns
            correspond to different weights in the linear layer."*/

        // Read and unpack the quantized collections
        int[] groupIndexes = readGroupIndexes(reader, id, rows);
        int[][] zeros = readZeros(gptqConfig, reader, id, rows, cols);
        Matrix scalesMatrix = readScales(gptqConfig, reader, id, rows, cols);
        int[][] quantizedWeights = readQuantizedWeights(gptqConfig, reader, id, rows, cols);

        // This is the collector of the de-quantized weights
        Matrix resultMatrix = emptyMatrix(config.getQuantizationConfig().getOutputFloatType(), rows, cols);

        for (var col = 0; col < cols; col++)
        {
            var x = 0;
            var groupIndex = 0;

            // Iterate over on the values within row
            for (var row = 0; row < rows; row++)
            {
                if (gptqConfig.getDescAct())
                {
                    groupIndex = groupIndexes[row];
                }
                else
                {
                    x++;
                    if (x > gptqConfig.getGroupSize())
                    {
                        groupIndex++;
                        x = 0;
                    }
                }

                int zero = zeros[groupIndex][col];

                float scale = scalesMatrix.getValue(groupIndex, col);

                int quantizedWeight = quantizedWeights[row][col];

                // This is the core GPTQ de-quantization algorithm:
                float weight = scale * (quantizedWeight - zero);

                resultMatrix.setValue(row, col, weight);
            }
        }


        // Iterate over on the rows
        /*var x = 0;
        var groupIndex = 0;
        for (var row = 0; row < rows; row++)
        {
            groupIndex = groupIndexes.get(row);

            //This was an attempt to not using the groupIndex:
            //x++;
            //if (x > groupSize)
            //{
            //    groupIndex++;
            //    x = 0;
            //}

            // Iterate over on the values within row
            for (var col = 0; col < cols; col++)
            {
                // At symmetric quantization the zero is always 0, otherwise use the stored value.
                int zero = gptqConfig.getSym() ? 1 : zeros[groupIndex][col] + 1;

                float scale = scalesMatrix.getValue(groupIndex, col);

                int quantizedValue = quantizedWeights[row][col];

                // This is the core GPTQ de-quantization algorithm:
                float weight = scale * (quantizedValue - zero);

                resultMatrix.setValue(row, col, weight);
            }
        }*/

        return resultMatrix;
    }

    private int[] readGroupIndexes(ParameterReader reader, String id, int rows)
    {
        // GroupIndex, int32 (Interestingly, isn't packed. A whole int32 is used to store few different values)
        return reader.readIntArray(getFinalParameterId(id, GROUP_INDEX_KEY), rows);
    }

    private int[][] readZeros(GptqConfig gptqConfig, ParameterReader reader, String id, int rows, int cols)
    {
        var bits = gptqConfig.getBits();
        var valuesPerInt32 = 32 / bits;
        var zerosRows = rows / gptqConfig.getGroupSize();
        var zerosCols = cols / valuesPerInt32;

        int[][] zerosMatrix = reader.readIntArray2D(getFinalParameterId(id, ZEROS_KEY), zerosRows, zerosCols);
        return unpack4bitsFromIntMatrixByRow(zerosMatrix); // TODO: Only 4 bits is supported here
    }

    private Matrix readScales(GptqConfig gptqConfig, ParameterReader reader, String id, int rows, int cols)
    {
        // Scales, FLOAT 16
        var scalesRows = rows / gptqConfig.getGroupSize();
        return reader.readFloat16Matrix(getFinalParameterId(id, SCALES_KEY), scalesRows, cols);
    }

    private int[][] readQuantizedWeights(GptqConfig gptqConfig, ParameterReader reader, String parameterId, int rows, int cols)
    {
        var bits = gptqConfig.getBits();
        var valuesPerInt32 = 32 / bits;
        if (bits == 2 || bits == 4 || bits == 8 || bits == 16)
        {
                /* "If bits is 2, 4, 8, or 16, the quantization method packs multiple weights
                   into a single 32-bit word using bit-shifting operations."*/
            // Quantized weights, packed ints
            var weightRows = rows / valuesPerInt32;
            int[][] quantizedWeightsMatrix = reader.readIntArray2D(getFinalParameterId(parameterId, WEIGHTS_KEY), weightRows, cols);

            return unpackIntMatrixByCol(quantizedWeightsMatrix, valuesPerInt32, bits);
        }
        else if (bits == 3)
        {
                /* "If bits is 3, the quantization method uses a more complex packing method
                   that packs 10 weights into 32 bits using a combination of bit-shifting and masking operations."*/
            // TODO
        }
        return null;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int size)
    {
        // TODO: Calculate
        return 0;
    }

    @Override
    public long calculateByteSize(ParameterReader reader, String parameterId, int rows, int cols)
    {
        // TODO: Calculate
        return 0;
    }
}
