package math;

import static java.lang.Math.PI;
import static java.lang.Math.tanh;

public interface NeuralNetUtil
{
    double SQRT_2_PER_PI = java.lang.Math.sqrt(2 / PI);

    default float relu(float value)
    {
        return value < 0 ? 0 : value;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    default float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(SQRT_2_PER_PI * (value + 0.044715 * value * value * value))));
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (faster approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    default float geluFast(float value)
    {
        return (float) (0.5 * value * (1 + tanh(value * 0.7978845608 * (0.044715 * value * value + 1))));
    }

    /**
     * SwiGLU activation function
     * Original paper: <a href="https://arxiv.org/abs/2002.05202" />
     */
    default float swiglu(float value)
    {
        return (float) (value * (1f / (1f + java.lang.Math.exp(-value))));
    }
}
