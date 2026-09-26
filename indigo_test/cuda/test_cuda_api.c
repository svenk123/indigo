// Tests device detection, context pool, device buffers and the self test of indigo_cuda.
// Without CUDA (make CUDA=no) the test only verifies that the stubs report "not available".

#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "indigo_cuda.h"

static int failures = 0;

#define CHECK(condition) do { if (!(condition)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; } } while (0)

#ifdef INDIGO_CUDA

#define THREADS 12
#define ITERATIONS 200

static int acquired_total = 0, refused_total = 0;
static pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *worker(void *arg) {
	(void)arg;
	for (int i = 0; i < ITERATIONS; i++) {
		indigo_cuda_context *context = indigo_cuda_acquire();
		if (context == NULL) { // pool exhausted: allowed, the caller falls back to the CPU
			pthread_mutex_lock(&counter_mutex);
			refused_total++;
			pthread_mutex_unlock(&counter_mutex);
			continue;
		}
		void *a = indigo_cuda_buffer(context, 0, 1 << 20);
		void *b = indigo_cuda_buffer(context, 1, (i % 7 + 1) << 16);
		CHECK(a != NULL && b != NULL && a != b);
		indigo_cuda_release(context);
		pthread_mutex_lock(&counter_mutex);
		acquired_total++;
		pthread_mutex_unlock(&counter_mutex);
	}
	return NULL;
}

int main() {
	if (!indigo_cuda_available()) {
		printf("SKIPPED: built with CUDA, but no usable device (or INDIGO_CUDA=0)\n");
		return 0;
	}
	printf("device: %s, compute capability %d\n", indigo_cuda_device_name(), indigo_cuda_compute_capability());
	CHECK(strlen(indigo_cuda_device_name()) > 0);
	CHECK(indigo_cuda_compute_capability() >= 50);
	CHECK(indigo_cuda_selftest());

	// buffers: valid slots, growing, invalid arguments
	indigo_cuda_context *context = indigo_cuda_acquire();
	CHECK(context != NULL);
	void *small = indigo_cuda_buffer(context, 0, 1024);
	CHECK(small != NULL);
	CHECK(indigo_cuda_buffer(context, 0, 512) == small); // reused, not reallocated
	CHECK(indigo_cuda_buffer(context, 0, 64 << 20) != NULL); // grows
	CHECK(indigo_cuda_buffer(context, -1, 16) == NULL);
	CHECK(indigo_cuda_buffer(context, INDIGO_CUDA_BUFFERS, 16) == NULL);
	CHECK(indigo_cuda_buffer(context, 0, 0) == NULL);
	CHECK(indigo_cuda_buffer(NULL, 0, 16) == NULL);
	indigo_cuda_release(context);
	indigo_cuda_release(NULL); // must not crash

	// concurrent use of the pool
	pthread_t threads[THREADS];
	for (int i = 0; i < THREADS; i++) {
		pthread_create(&threads[i], NULL, worker, NULL);
	}
	for (int i = 0; i < THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	CHECK(acquired_total > 0);
	CHECK(acquired_total + refused_total == THREADS * ITERATIONS);
	printf("pool: %d acquired, %d refused (pool exhausted)\n", acquired_total, refused_total);
	CHECK(indigo_cuda_selftest()); // still works after the stress test

	printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
	return failures ? 1 : 0;
}

#else

int main() {
	CHECK(!indigo_cuda_available());
	CHECK(indigo_cuda_acquire() == NULL);
	CHECK(indigo_cuda_buffer(NULL, 0, 16) == NULL);
	CHECK(!indigo_cuda_selftest());
	CHECK(indigo_cuda_compute_capability() == 0);
	printf("%s (built without CUDA, stubs only): %d failures\n", failures ? "FAILED" : "PASSED", failures);
	return failures ? 1 : 0;
}

#endif
