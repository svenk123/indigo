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

// GPU acceleration of indigo_equalize_bayer_channels() (the full-frame scaling pass) and indigo_binarize()
// from indigo_raw_utils.c. Both reduce to exact integer sums (order independent, so the result is bit-identical
// to the CPU code regardless of how the GPU schedules threads) followed by an independent per-pixel map.

#include <cuda_runtime.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>

#include "indigo_cuda.h"

static bool cuda_ok(cudaError_t error, const char *what) {
	if (error == cudaSuccess) {
		return true;
	}
	indigo_error("CUDA: %s failed: %s", what, cudaGetErrorString(error));
	return false;
}

//---------------------------------------------------------------------
// indigo_equalize_bayer_channels: apply the scale factors to the whole frame
//---------------------------------------------------------------------

// one thread per 2x2 block, exactly like the CPU loop (step 2 in x and y)
template <typename T> __global__ static void scale_bayer_kernel(T *data, int width, int height, double s1, double s2, double s3, double s4) {
	const int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	const int y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width - 1 || y >= height - 1) {
		return;
	}
	const int index = y * width + x;
	data[index] = (T)(data[index] * s1);
	data[index + 1] = (T)(data[index + 1] * s3);
	data[index + width] = (T)(data[index + width] * s2);
	data[index + width + 1] = (T)(data[index + width + 1] * s4);
}

template <typename T> static bool scale_bayer(T *data, int width, int height, double s1, double s2, double s3, double s4) {
	if ((size_t)width * height < INDIGO_CUDA_MIN_PIXELS || !indigo_cuda_available()) {
		return false;
	}
	indigo_cuda_context *context = indigo_cuda_acquire();
	if (context == NULL) {
		return false;
	}
	bool done = false;
	const size_t bytes = (size_t)width * height * sizeof(T);
	T *device_data = (T *)indigo_cuda_buffer(context, 0, bytes);
	cudaStream_t stream = (cudaStream_t)indigo_cuda_stream(context);
	if (device_data) {
		done = cuda_ok(cudaMemcpyAsync(device_data, data, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync (upload)");
		if (done) {
			const dim3 block(32, 8);
			const dim3 grid((width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);
			scale_bayer_kernel<T><<<grid, block, 0, stream>>>(device_data, width, height, s1, s2, s3, s4);
			done = cuda_ok(cudaGetLastError(), "kernel launch");
		}
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(data, device_data, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (download)");
		}
		if (done) {
			done = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
		} else {
			cudaStreamSynchronize(stream);
		}
	}
	indigo_cuda_release(context);
	return done;
}

extern "C" bool indigo_cuda_equalize_bayer_scale(indigo_raw_type raw_type, void *data, int width, int height, double ch1_scale, double ch2_scale, double ch3_scale, double ch4_scale) {
	if (raw_type == INDIGO_RAW_MONO16) {
		return scale_bayer<uint16_t>((uint16_t *)data, width, height, ch1_scale, ch2_scale, ch3_scale, ch4_scale);
	} else if (raw_type == INDIGO_RAW_MONO8) {
		return scale_bayer<uint8_t>((uint8_t *)data, width, height, ch1_scale, ch2_scale, ch3_scale, ch4_scale);
	}
	return false;
}

//---------------------------------------------------------------------
// indigo_binarize
//---------------------------------------------------------------------

#define BINARIZE_BLOCK 256

// per pixel luminance, matching the switch in indigo_binarize() exactly (data8/data16 point at the same buffer)
template <int RAW_TYPE> __device__ static inline int binarize_luminance(const uint8_t *data8, const uint16_t *data16, int i) {
	if (RAW_TYPE == INDIGO_RAW_MONO8) {
		return data8[i];
	} else if (RAW_TYPE == INDIGO_RAW_MONO16) {
		return data16[i];
	} else if (RAW_TYPE == INDIGO_RAW_RGB24) {
		const int i3 = 3 * i;
		return (data8[i3] + data8[i3 + 1] + data8[i3 + 2]) / 3;
	} else if (RAW_TYPE == INDIGO_RAW_RGBA32) {
		const int i4 = 4 * i;
		return (data8[i4] + data8[i4 + 1] + data8[i4 + 2]) / 3;
	} else if (RAW_TYPE == INDIGO_RAW_ABGR32) {
		const int i4 = 4 * i;
		return (data8[i4 + 1] + data8[i4 + 2] + data8[i4 + 3]) / 3;
	} else { // INDIGO_RAW_RGB48
		const int i3 = 3 * i;
		return (data16[i3] + data16[i3 + 1] + data16[i3 + 2]) / 3;
	}
}

template <int RAW_TYPE> __global__ static void binarize_reduce_kernel(const uint8_t *data8, const uint16_t *data16, int size, unsigned long long *sum_out, unsigned long long *sum_sq_out) {
	__shared__ unsigned long long partial_sum[BINARIZE_BLOCK];
	__shared__ unsigned long long partial_sum_sq[BINARIZE_BLOCK];
	unsigned long long sum = 0, sum_sq = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
		const unsigned int value = binarize_luminance<RAW_TYPE>(data8, data16, i);
		sum += value;
		sum_sq += (unsigned long long)value * value;
	}
	partial_sum[threadIdx.x] = sum;
	partial_sum_sq[threadIdx.x] = sum_sq;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
			partial_sum_sq[threadIdx.x] += partial_sum_sq[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(sum_out, partial_sum[0]);
		atomicAdd(sum_sq_out, partial_sum_sq[0]);
	}
}

// also counts the white (> threshold) pixels, like the CPU code's "sum" - indigo_binarize() rejects the result
// (too many bright pixels usually means the threshold estimate broke down) if more than 10% of pixels are white
// CPU comparison value for the threshold pass - the RGB types compare the *sum* of the channels against
// threshold * 3 instead of the averaged binarize_luminance() against threshold, which avoids the rounding of the
// integer division by 3 (see indigo_binarize()). MONO8/MONO16 use the plain value, same as binarize_luminance().
template <int RAW_TYPE> __device__ static inline int binarize_threshold_value(const uint8_t *data8, const uint16_t *data16, int i) {
	if (RAW_TYPE == INDIGO_RAW_MONO8) {
		return data8[i];
	} else if (RAW_TYPE == INDIGO_RAW_MONO16) {
		return data16[i];
	} else if (RAW_TYPE == INDIGO_RAW_RGB24) {
		const int i3 = 3 * i;
		return data8[i3] + data8[i3 + 1] + data8[i3 + 2];
	} else if (RAW_TYPE == INDIGO_RAW_RGBA32) {
		const int i4 = 4 * i;
		return data8[i4] + data8[i4 + 1] + data8[i4 + 2];
	} else if (RAW_TYPE == INDIGO_RAW_ABGR32) {
		const int i4 = 4 * i;
		return data8[i4 + 1] + data8[i4 + 2] + data8[i4 + 3];
	} else { // INDIGO_RAW_RGB48
		const int i3 = 3 * i;
		return data16[i3] + data16[i3 + 1] + data16[i3 + 2];
	}
}

template <int RAW_TYPE> __global__ static void binarize_threshold_kernel(const uint8_t *data8, const uint16_t *data16, int size, int threshold, uint8_t *output, unsigned long long *white_count_out) {
	__shared__ unsigned long long partial_count[BINARIZE_BLOCK];
	unsigned long long white_count = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
		const bool white = binarize_threshold_value<RAW_TYPE>(data8, data16, i) > threshold;
		output[i] = white ? 255 : 0;
		white_count += white;
	}
	partial_count[threadIdx.x] = white_count;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			partial_count[threadIdx.x] += partial_count[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(white_count_out, partial_count[0]);
	}
}

template <int RAW_TYPE> static bool binarize(const void *data, int width, int height, double sigma, uint8_t *output) {
	const int size = width * height;
	if ((size_t)size < INDIGO_CUDA_MIN_PIXELS || !indigo_cuda_available()) {
		return false;
	}
	const int elements = RAW_TYPE == INDIGO_RAW_RGB24 || RAW_TYPE == INDIGO_RAW_RGB48 ? 3 * size : (RAW_TYPE == INDIGO_RAW_RGBA32 || RAW_TYPE == INDIGO_RAW_ABGR32 ? 4 * size : size);
	const bool wide = RAW_TYPE == INDIGO_RAW_MONO16 || RAW_TYPE == INDIGO_RAW_RGB48;
	const size_t input_bytes = (size_t)elements * (wide ? sizeof(uint16_t) : sizeof(uint8_t));
	indigo_cuda_context *context = indigo_cuda_acquire();
	if (context == NULL) {
		return false;
	}
	bool done = false;
	uint8_t *device_input = (uint8_t *)indigo_cuda_buffer(context, 0, input_bytes);
	uint8_t *device_output = (uint8_t *)indigo_cuda_buffer(context, 1, (size_t)size);
	unsigned long long *device_counters = (unsigned long long *)indigo_cuda_buffer(context, 2, 2 * sizeof(unsigned long long));
	cudaStream_t stream = (cudaStream_t)indigo_cuda_stream(context);
	if (device_input && device_output && device_counters) {
		done = cuda_ok(cudaMemsetAsync(device_counters, 0, 2 * sizeof(unsigned long long), stream), "cudaMemsetAsync");
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(device_input, data, input_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync (upload)");
		}
		const uint8_t *data8 = wide ? NULL : device_input;
		const uint16_t *data16 = wide ? (uint16_t *)device_input : NULL;
		unsigned long long counters[2] = { 0, 0 };
		if (done) {
			const int blocks = (int)((size + BINARIZE_BLOCK - 1) / BINARIZE_BLOCK < 1024 ? (size + BINARIZE_BLOCK - 1) / BINARIZE_BLOCK : 1024);
			binarize_reduce_kernel<RAW_TYPE><<<blocks, BINARIZE_BLOCK, 0, stream>>>(data8, data16, size, device_counters, device_counters + 1);
			done = cuda_ok(cudaGetLastError(), "kernel launch (reduce)");
		}
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(counters, device_counters, sizeof(counters), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (counters)");
		}
		if (done) {
			done = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize (reduce)");
		}
		unsigned long long white_count = 0;
		if (done) {
			// same arithmetic as the CPU code: mean/stddev in double from exact integer sums, threshold truncated to int
			const double mean = (double)counters[0] / size;
			const double stddev = sqrt((double)counters[1] / size - mean * mean);
			int threshold = (int)(mean + sigma * stddev);
			if (RAW_TYPE == INDIGO_RAW_RGB24 || RAW_TYPE == INDIGO_RAW_RGBA32 || RAW_TYPE == INDIGO_RAW_ABGR32 || RAW_TYPE == INDIGO_RAW_RGB48) {
				threshold *= 3; // binarize_threshold_value() returns the sum of the 3 channels, not their average
			}
			done = cuda_ok(cudaMemsetAsync(device_counters, 0, sizeof(unsigned long long), stream), "cudaMemsetAsync (white count)");
			if (done) {
				const int blocks = (int)((size + BINARIZE_BLOCK - 1) / BINARIZE_BLOCK < 1024 ? (size + BINARIZE_BLOCK - 1) / BINARIZE_BLOCK : 1024);
				binarize_threshold_kernel<RAW_TYPE><<<blocks, BINARIZE_BLOCK, 0, stream>>>(data8, data16, size, threshold, device_output, device_counters);
				done = cuda_ok(cudaGetLastError(), "kernel launch (threshold)");
			}
		}
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(&white_count, device_counters, sizeof(white_count), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (white count)");
		}
		if (done) {
			done = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
		} else {
			cudaStreamSynchronize(stream);
		}
		// same safety check as the CPU code: too many bright pixels usually means the threshold estimate broke down
		if (done && white_count > size * 0.1) {
			done = false;
		}
		if (done) {
			done = cuda_ok(cudaMemcpyAsync(output, device_output, (size_t)size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (download)");
			if (done) {
				done = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize (download)");
			}
		}
	}
	indigo_cuda_release(context);
	return done;
}

extern "C" bool indigo_cuda_binarize(indigo_raw_type raw_type, const void *data, int width, int height, double sigma, uint8_t *output) {
	switch (raw_type) {
		case INDIGO_RAW_MONO8: return binarize<INDIGO_RAW_MONO8>(data, width, height, sigma, output);
		case INDIGO_RAW_MONO16: return binarize<INDIGO_RAW_MONO16>(data, width, height, sigma, output);
		case INDIGO_RAW_RGB24: return binarize<INDIGO_RAW_RGB24>(data, width, height, sigma, output);
		case INDIGO_RAW_RGBA32: return binarize<INDIGO_RAW_RGBA32>(data, width, height, sigma, output);
		case INDIGO_RAW_ABGR32: return binarize<INDIGO_RAW_ABGR32>(data, width, height, sigma, output);
		case INDIGO_RAW_RGB48: return binarize<INDIGO_RAW_RGB48>(data, width, height, sigma, output);
		default: return false;
	}
}
