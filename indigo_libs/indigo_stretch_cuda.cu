// You can use this software under the terms of 'INDIGO Astronomy
// open-source license' (see LICENSE.md).
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS 'AS IS' AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// GPU version of the preview stretch of indigo_stretch.cpp. The pixel functions are shared with the CPU code
// (indigo_stretch_kernels.h), the frame is processed pixel by pixel, so the results are identical.
// Stretch parameters (median, MAD) are still computed on the CPU, they take a few milliseconds.

#include <cuda_runtime.h>

#include <indigo/indigo_bus.h>

#include "indigo_cuda.h"
#include "indigo_stretch_kernels.h"

typedef struct {
	indigo_stretch_channel channel[3];
} stretch_channels;

template <typename T> __global__ static void debayer_stretch_kernel(const T *input, uint8_t *output, int width, int height, int offsets, stretch_channels channels) {
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (column >= width || row >= height) {
		return;
	}
	const int index = row * width + column;
	float red = 0, green = 0, blue = 0;
	indigo_debayer_pixel(input, index, row, column, width, height, offsets, red, green, blue);
	uint8_t *pixel = output + 3 * (size_t)index;
	pixel[0] = indigo_stretch_value(red, channels.channel[0].native_shadows, channels.channel[0].native_highlights, channels.channel[0].k1_k2, channels.channel[0].midtones_k2);
	pixel[1] = indigo_stretch_value(green, channels.channel[1].native_shadows, channels.channel[1].native_highlights, channels.channel[1].k1_k2, channels.channel[1].midtones_k2);
	pixel[2] = indigo_stretch_value(blue, channels.channel[2].native_shadows, channels.channel[2].native_highlights, channels.channel[2].k1_k2, channels.channel[2].midtones_k2);
}

// libjpeg (jccolor.c) fixed point RGB -> YCbCr, the same constants and rounding
#define SCALEBITS 16
#define CBCR_OFFSET ((int)128 << SCALEBITS)
#define ONE_HALF ((int)1 << (SCALEBITS - 1))
#define FIX(x) ((int)((x) * (1L << SCALEBITS) + 0.5))

__device__ static inline void rgb_to_ycc(int r, int g, int b, int &y, int &cb, int &cr) {
	y = (FIX(0.299) * r + FIX(0.587) * g + FIX(0.114) * b + ONE_HALF) >> SCALEBITS;
	cb = ((-FIX(0.168735892)) * r + (-FIX(0.331264108)) * g + FIX(0.5) * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS;
	cr = (FIX(0.5) * r + (-FIX(0.418687589)) * g + (-FIX(0.081312411)) * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS;
}

// One thread per 2x2 block: debayer + stretch of the four pixels (the same for all threads of a warp, so no divergence),
// Y for each pixel, Cb and Cr averaged like libjpeg h2v2_downsample() does (bias 1, 2, 1, 2 ... per column).
// Padding to multiples of 16 is done like libjpeg does it: columns behind the frame replicate the last column before
// the chroma is averaged, rows behind the frame replicate the last row of the Y plane and the last row of the (already
// averaged) Cb and Cr planes.
template <typename T> __global__ static void debayer_stretch_ycc_kernel(const T *input, uint8_t *y_plane, uint8_t *cb_plane, uint8_t *cr_plane, int width, int height, int offsets, stretch_channels channels, int y_stride, int c_stride, int c_rows) {
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	if (cx >= c_stride || cy >= c_rows) {
		return;
	}
	const int last_c_row = (height - 1) / 2;
	const bool padding = cy > last_c_row; // chroma row completely behind the frame
	const int block_row = padding ? last_c_row : cy;
	int cb_sum = 0, cr_sum = 0, y_last[2] = { 0, 0 };
	#pragma unroll
	for (int dy = 0; dy < 2; dy++) {
		#pragma unroll
		for (int dx = 0; dx < 2; dx++) {
			const int column = min(2 * cx + dx, width - 1);
			const int row = min(2 * block_row + dy, height - 1);
			float red = 0, green = 0, blue = 0;
			indigo_debayer_pixel(input, row * width + column, row, column, width, height, offsets, red, green, blue);
			const int r = indigo_stretch_value(red, channels.channel[0].native_shadows, channels.channel[0].native_highlights, channels.channel[0].k1_k2, channels.channel[0].midtones_k2);
			const int g = indigo_stretch_value(green, channels.channel[1].native_shadows, channels.channel[1].native_highlights, channels.channel[1].k1_k2, channels.channel[1].midtones_k2);
			const int b = indigo_stretch_value(blue, channels.channel[2].native_shadows, channels.channel[2].native_highlights, channels.channel[2].k1_k2, channels.channel[2].midtones_k2);
			int y, cb, cr;
			rgb_to_ycc(r, g, b, y, cb, cr);
			if (dy == 1) {
				y_last[dx] = y; // the last row of the block is the last row of the frame if this is a padding block
			}
			if (!padding) {
				y_plane[(size_t)(2 * cy + dy) * y_stride + 2 * cx + dx] = y;
			}
			cb_sum += cb;
			cr_sum += cr;
		}
	}
	if (padding) {
		#pragma unroll
		for (int dx = 0; dx < 2; dx++) {
			y_plane[(size_t)(2 * cy) * y_stride + 2 * cx + dx] = y_last[dx];
			y_plane[(size_t)(2 * cy + 1) * y_stride + 2 * cx + dx] = y_last[dx];
		}
	}
	const int bias = (cx & 1) ? 2 : 1;
	cb_plane[(size_t)cy * c_stride + cx] = (cb_sum + bias) >> 2;
	cr_plane[(size_t)cy * c_stride + cx] = (cr_sum + bias) >> 2;
}

template <typename T> __global__ static void debayer_kernel(const T *input, uint8_t *output, int width, int height, int offsets) {
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (column >= width || row >= height) {
		return;
	}
	const int index = row * width + column;
	float red = 0, green = 0, blue = 0;
	indigo_debayer_pixel(input, index, row, column, width, height, offsets, red, green, blue);
	uint8_t *pixel = output + 3 * (size_t)index;
	pixel[0] = red;
	pixel[1] = green;
	pixel[2] = blue;
}

// mono: one channel, coef 1
template <typename T> __global__ static void stretch_kernel(const T *input, uint8_t *output, size_t size, indigo_stretch_channel channel) {
	const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		output[index] = indigo_stretch_value(input[index] / 1.0f, channel.native_shadows, channel.native_highlights, channel.k1_k2, channel.midtones_k2);
	}
}

// interleaved RGB: the three components share the stretch, each has its own coefficient
template <typename T> __global__ static void stretch_rgb_kernel(const T *input, uint8_t *output, size_t size, indigo_stretch_channel channel, float coef0, float coef1, float coef2) {
	const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const float coef[3] = { coef0, coef1, coef2 };
		for (int i = 0; i < 3; i++) {
			output[3 * index + i] = indigo_stretch_value(input[3 * index + i] / coef[i], channel.native_shadows, channel.native_highlights, channel.k1_k2, channel.midtones_k2);
		}
	}
}

static volatile unsigned long completed_calls = 0;

extern "C" unsigned long indigo_cuda_stretch_calls(void) {
	return completed_calls;
}

static bool cuda_ok(cudaError_t error, const char *what) {
	if (error == cudaSuccess) {
		return true;
	}
	indigo_error("CUDA: %s failed: %s", what, cudaGetErrorString(error));
	return false;
}

// Uploads the frame, runs launch(context, device_input, device_output), downloads the result.
// Returns false if the GPU is not used, the output is then untouched or incomplete and the CPU code has to run.
template <typename T, typename Launch> static bool run(const T *input, size_t input_elements, uint8_t *output, size_t output_bytes, int width, int height, Launch launch) {
	if ((size_t)width * height < INDIGO_CUDA_MIN_PIXELS || !indigo_cuda_available()) {
		return false;
	}
	indigo_cuda_context *context = indigo_cuda_acquire();
	if (context == NULL) {
		return false;
	}
	bool done = false;
	const size_t input_bytes = input_elements * sizeof(T);
	T *device_input = (T *)indigo_cuda_buffer(context, 0, input_bytes);
	uint8_t *device_output = (uint8_t *)indigo_cuda_buffer(context, 1, output_bytes);
	cudaStream_t stream = (cudaStream_t)indigo_cuda_stream(context);
	if (device_input && device_output) {
		done = cuda_ok(cudaMemcpyAsync(device_input, input, input_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync (upload)");
		if (done) {
			launch(device_input, device_output, stream);
			done = cuda_ok(cudaGetLastError(), "kernel launch");
		}
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(output, device_output, output_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (download)");
		}
		if (done) {
			done = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
		} else {
			cudaStreamSynchronize(stream);
		}
	}
	indigo_cuda_release(context);
	if (done) {
		__sync_fetch_and_add(&completed_calls, 1);
	}
	return done;
}

static stretch_channels make_channels(const double *shadows, const double *midtones, const double *highlights, double max_input) {
	stretch_channels channels;
	for (int i = 0; i < 3; i++) {
		channels.channel[i] = indigo_make_stretch_channel(shadows[i], midtones[i], highlights[i], max_input);
	}
	return channels;
}

static dim3 block_2d() {
	return dim3(32, 8);
}

static dim3 grid_2d(int width, int height) {
	return dim3((width + 31) / 32, (height + 7) / 8);
}

template <typename T> static bool debayer_stretch(const T *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights) {
	const stretch_channels channels = make_channels(shadows, midtones, highlights, sizeof(T) == 1 ? 0xFF : 0xFFFF);
	return run<T>(input, (size_t)width * height, output, (size_t)width * height * 3, width, height, [&](T *device_input, uint8_t *device_output, cudaStream_t stream) {
		debayer_stretch_kernel<T><<<grid_2d(width, height), block_2d(), 0, stream>>>(device_input, device_output, width, height, offsets, channels);
	});
}

template <typename T> static bool debayer_stretch_ycc(const T *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights) {
	const stretch_channels channels = make_channels(shadows, midtones, highlights, sizeof(T) == 1 ? 0xFF : 0xFFFF);
	const indigo_ycc420_layout layout = indigo_ycc420_layout_for(width, height);
	return run<T>(input, (size_t)width * height, planes, layout.y_size + 2 * layout.c_size, width, height, [&](T *device_input, uint8_t *device_output, cudaStream_t stream) {
		uint8_t *y_plane = device_output, *cb_plane = y_plane + layout.y_size, *cr_plane = cb_plane + layout.c_size;
		debayer_stretch_ycc_kernel<T><<<grid_2d(layout.c_stride, layout.c_rows), block_2d(), 0, stream>>>(device_input, y_plane, cb_plane, cr_plane, width, height, offsets, channels, layout.y_stride, layout.c_stride, layout.c_rows);
	});
}

template <typename T> static bool stretch_mono(const T *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights) {
	const indigo_stretch_channel channel = indigo_make_stretch_channel(shadows, midtones, highlights, sizeof(T) == 1 ? 0xFF : 0xFFFF);
	const size_t size = (size_t)width * height;
	return run<T>(input, size, output, size, width, height, [&](T *device_input, uint8_t *device_output, cudaStream_t stream) {
		stretch_kernel<T><<<(unsigned)((size + 255) / 256), 256, 0, stream>>>(device_input, device_output, size, channel);
	});
}

template <typename T> static bool stretch_rgb(const T *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef) {
	const indigo_stretch_channel channel = indigo_make_stretch_channel(shadows, midtones, highlights, sizeof(T) == 1 ? 0xFF : 0xFFFF);
	const size_t size = (size_t)width * height;
	return run<T>(input, size * 3, output, size * 3, width, height, [&](T *device_input, uint8_t *device_output, cudaStream_t stream) {
		stretch_rgb_kernel<T><<<(unsigned)((size + 255) / 256), 256, 0, stream>>>(device_input, device_output, size, channel, coef[0], coef[1], coef[2]);
	});
}

extern "C" bool indigo_cuda_debayer_stretch_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights) {
	return debayer_stretch<uint8_t>(input, width, height, offsets, output, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_debayer_stretch_16(const uint16_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights) {
	return debayer_stretch<uint16_t>(input, width, height, offsets, output, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_debayer_stretch_ycc_8(const uint8_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights) {
	return debayer_stretch_ycc<uint8_t>(input, width, height, offsets, planes, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_debayer_stretch_ycc_16(const uint16_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights) {
	return debayer_stretch_ycc<uint16_t>(input, width, height, offsets, planes, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_debayer_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output) {
	return run<uint8_t>(input, (size_t)width * height, output, (size_t)width * height * 3, width, height, [&](uint8_t *device_input, uint8_t *device_output, cudaStream_t stream) {
		debayer_kernel<uint8_t><<<grid_2d(width, height), block_2d(), 0, stream>>>(device_input, device_output, width, height, offsets);
	});
}

extern "C" bool indigo_cuda_stretch_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights) {
	return stretch_mono<uint8_t>(input, width, height, output, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_stretch_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights) {
	return stretch_mono<uint16_t>(input, width, height, output, shadows, midtones, highlights);
}

extern "C" bool indigo_cuda_stretch_rgb_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef) {
	return stretch_rgb<uint8_t>(input, width, height, output, shadows, midtones, highlights, coef);
}

extern "C" bool indigo_cuda_stretch_rgb_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef) {
	return stretch_rgb<uint16_t>(input, width, height, output, shadows, midtones, highlights, coef);
}
