//
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

// Device detection, context pool and self test of the optional CUDA acceleration.
// Must stay compilable with the CUDA toolkits of all supported JetPack releases (C++11).

#include <atomic>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <cuda_runtime.h>

#include <indigo/indigo_bus.h>

#include "indigo_cuda.h"

#define INDIGO_CUDA_MAX_CONTEXTS 8

struct indigo_cuda_context {
	cudaStream_t stream;
	void *buffer[INDIGO_CUDA_BUFFERS];
	size_t buffer_size[INDIGO_CUDA_BUFFERS];
	bool in_use;
};

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static int state = 0; // 0 - not checked yet, 1 - available, -1 - not available
static std::atomic<bool> enabled(true);
static char device_name[256] = "";
static int compute_capability = 0;
static indigo_cuda_context *contexts[INDIGO_CUDA_MAX_CONTEXTS];

static bool cuda_ok(cudaError_t error, const char *what) {
	if (error == cudaSuccess) {
		return true;
	}
	indigo_error("CUDA: %s failed: %s", what, cudaGetErrorString(error));
	return false;
}

static bool disabled_by_environment(void) {
	const char *value = getenv("INDIGO_CUDA");
	return value && (!strcmp(value, "0") || !strcasecmp(value, "off") || !strcasecmp(value, "no") || !strcasecmp(value, "false"));
}

__global__ static void selftest_kernel(int *data) {
	data[threadIdx.x] = 2 * threadIdx.x;
}

// caller holds the mutex
static void detect(void) {
	state = -1;
	if (disabled_by_environment()) {
		indigo_log("CUDA: disabled by environment variable INDIGO_CUDA");
		return;
	}
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if (error != cudaSuccess || count < 1) {
		indigo_log("CUDA: no device available (%s)", error == cudaSuccess ? "device count is 0" : cudaGetErrorString(error));
		return;
	}
	cudaDeviceProp properties;
	if (!cuda_ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
		return;
	}
	// the binary contains machine code for the architectures chosen at build time only - check that a kernel can be loaded
	cudaFuncAttributes attributes;
	error = cudaFuncGetAttributes(&attributes, selftest_kernel);
	if (error != cudaSuccess) {
		indigo_error("CUDA: %s (compute capability %d.%d) is not supported by this build: %s", properties.name, properties.major, properties.minor, cudaGetErrorString(error));
		return;
	}
	strncpy(device_name, properties.name, sizeof(device_name) - 1);
	compute_capability = properties.major * 10 + properties.minor;
	state = 1;
	indigo_log("CUDA: %s, compute capability %d.%d, %d MB", device_name, properties.major, properties.minor, (int)(properties.totalGlobalMem >> 20));
}

extern "C" void indigo_cuda_set_enabled(bool value) {
	enabled = value;
}

extern "C" bool indigo_cuda_available(void) {
	if (!enabled) {
		return false;
	}
	pthread_mutex_lock(&mutex);
	if (state == 0) {
		detect();
	}
	bool available = state == 1;
	pthread_mutex_unlock(&mutex);
	return available;
}

extern "C" const char *indigo_cuda_device_name(void) {
	return indigo_cuda_available() ? device_name : "";
}

extern "C" int indigo_cuda_compute_capability(void) {
	return indigo_cuda_available() ? compute_capability : 0;
}

extern "C" indigo_cuda_context *indigo_cuda_acquire(void) {
	if (!indigo_cuda_available()) {
		return NULL;
	}
	indigo_cuda_context *context = NULL;
	pthread_mutex_lock(&mutex);
	for (int i = 0; i < INDIGO_CUDA_MAX_CONTEXTS && context == NULL; i++) {
		if (contexts[i] == NULL) {
			indigo_cuda_context *created = (indigo_cuda_context *)calloc(1, sizeof(indigo_cuda_context));
			if (created == NULL || !cuda_ok(cudaStreamCreateWithFlags(&created->stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
				free(created);
				break;
			}
			contexts[i] = created;
		}
		if (!contexts[i]->in_use) {
			contexts[i]->in_use = true;
			context = contexts[i];
		}
	}
	pthread_mutex_unlock(&mutex);
	if (context == NULL) {
		indigo_debug("CUDA: no free context, CPU fallback");
	}
	return context;
}

extern "C" void indigo_cuda_release(indigo_cuda_context *context) {
	if (context == NULL) {
		return;
	}
	pthread_mutex_lock(&mutex);
	context->in_use = false;
	pthread_mutex_unlock(&mutex);
}

extern "C" void *indigo_cuda_stream(indigo_cuda_context *context) {
	return context ? (void *)context->stream : NULL;
}

extern "C" void *indigo_cuda_buffer(indigo_cuda_context *context, int slot, size_t size) {
	if (context == NULL || slot < 0 || slot >= INDIGO_CUDA_BUFFERS || size == 0) {
		return NULL;
	}
	if (context->buffer_size[slot] < size) {
		if (context->buffer[slot]) {
			cudaFree(context->buffer[slot]);
			context->buffer[slot] = NULL;
			context->buffer_size[slot] = 0;
		}
		void *pointer = NULL;
		if (!cuda_ok(cudaMalloc(&pointer, size), "cudaMalloc")) {
			return NULL;
		}
		context->buffer[slot] = pointer;
		context->buffer_size[slot] = size;
	}
	return context->buffer[slot];
}

extern "C" bool indigo_cuda_selftest(void) {
	indigo_cuda_context *context = indigo_cuda_acquire();
	if (context == NULL) {
		return false;
	}
	bool result = false;
	int host[32];
	int *device = (int *)indigo_cuda_buffer(context, 0, sizeof(host));
	if (device) {
		selftest_kernel<<<1, 32, 0, context->stream>>>(device);
		if (cuda_ok(cudaGetLastError(), "kernel launch") && cuda_ok(cudaMemcpyAsync(host, device, sizeof(host), cudaMemcpyDeviceToHost, context->stream), "cudaMemcpyAsync") && cuda_ok(cudaStreamSynchronize(context->stream), "cudaStreamSynchronize")) {
			result = true;
			for (int i = 0; i < 32; i++) {
				if (host[i] != 2 * i) {
					indigo_error("CUDA: self test returned a wrong result");
					result = false;
					break;
				}
			}
		}
	}
	indigo_cuda_release(context);
	return result;
}
