// Compares indigo_binarize() and indigo_equalize_bayer_channels() on the GPU with the CPU code (switched with
// indigo_cuda_set_enabled): the results must be byte-identical (both reduce to exact integer sums, order
// independent, see indigo_raw_utils_cuda.cu). Frames below INDIGO_CUDA_MIN_PIXELS stay on the CPU (checked too).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>
#include <indigo/indigo_bahtinov.h>

#include "indigo_cuda.h"
#include "test_image.h"

static int checks = 0, failures = 0;

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

// there is no separate CUDA call counter for these two functions (unlike the stretch), so "was the GPU used" is
// inferred from indigo_cuda_available() and the pixel count, exactly like the library itself decides
static bool gpu_expected(size_t pixels) {
	return indigo_cuda_available() && pixels >= INDIGO_CUDA_MIN_PIXELS;
}

static void check_binarize(const char *label, indigo_raw_type type, const void *data, int width, int height, double sigma, bool timing) {
	const size_t size = (size_t)width * height;
	indigo_cuda_set_enabled(false);
	uint8_t *cpu = indigo_binarize(type, data, width, height, sigma);
	indigo_cuda_set_enabled(true);
	uint8_t *gpu = indigo_binarize(type, data, width, height, sigma);
	checks++;
	// indigo_binarize() returns NULL if more than 10% of pixels are white (broken threshold estimate) - both sides
	// must agree on that too, not just on the pixel data
	const bool ok = (cpu == NULL) == (gpu == NULL) && (cpu == NULL || memcmp(cpu, gpu, size) == 0);
	if (!ok) failures++;
	printf("%s binarize %-22s %5dx%-5d%s%s\n", ok ? "ok  " : "FAIL", label, width, height, cpu == NULL ? " (rejected: too many bright pixels)" : "", ok ? "" : ", DIFFERS from the CPU result");
	if (timing) {
		double t_cpu[5], t_gpu[5];
		for (int r = 0; r < 5; r++) {
			indigo_cuda_set_enabled(false);
			double t = now_ms(); free(indigo_binarize(type, data, width, height, sigma)); t_cpu[r] = now_ms() - t;
			indigo_cuda_set_enabled(true);
			t = now_ms(); free(indigo_binarize(type, data, width, height, sigma)); t_gpu[r] = now_ms() - t;
		}
		qsort(t_cpu, 5, sizeof(double), compare_double);
		qsort(t_gpu, 5, sizeof(double), compare_double);
		printf("     timing: CPU %.1f ms, GPU %.1f ms\n", t_cpu[2], t_gpu[2]);
	}
	free(cpu);
	free(gpu);
}

static void check_equalize(const char *label, indigo_raw_type type, const void *data, int width, int height, bool timing) {
	const size_t size = (size_t)width * height * (type == INDIGO_RAW_MONO16 ? 2 : 1);
	void *cpu = malloc(size);
	void *gpu = malloc(size);
	memcpy(cpu, data, size);
	memcpy(gpu, data, size);
	indigo_cuda_set_enabled(false);
	indigo_result r1 = indigo_equalize_bayer_channels(type, cpu, width, height);
	indigo_cuda_set_enabled(true);
	indigo_result r2 = indigo_equalize_bayer_channels(type, gpu, width, height);
	checks++;
	const bool ok = r1 == r2 && (r1 != INDIGO_OK || memcmp(cpu, gpu, size) == 0);
	if (!ok) failures++;
	printf("%s equalize_bayer %-16s %5dx%-5d%s\n", ok ? "ok  " : "FAIL", label, width, height, ok ? "" : ", DIFFERS from the CPU result");
	if (timing) {
		double t_cpu[5], t_gpu[5];
		for (int r = 0; r < 5; r++) {
			memcpy(cpu, data, size);
			indigo_cuda_set_enabled(false);
			double t = now_ms(); indigo_equalize_bayer_channels(type, cpu, width, height); t_cpu[r] = now_ms() - t;
			memcpy(gpu, data, size);
			indigo_cuda_set_enabled(true);
			t = now_ms(); indigo_equalize_bayer_channels(type, gpu, width, height); t_gpu[r] = now_ms() - t;
		}
		qsort(t_cpu, 5, sizeof(double), compare_double);
		qsort(t_gpu, 5, sizeof(double), compare_double);
		printf("     timing: CPU %.1f ms, GPU %.1f ms\n", t_cpu[2], t_gpu[2]);
	}
	free(cpu);
	free(gpu);
}

int main() {
	if (!indigo_cuda_available()) {
		printf("SKIPPED: no CUDA device or built without CUDA\n");
		return 0;
	}
	printf("device: %s\n", indigo_cuda_device_name());
	const int sizes[][2] = { { 4, 4 }, { 301, 203 }, { 640, 481 }, { 1001, 803 }, { 2049, 1027 }, { 4144, 2822 } };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		const int w = sizes[s][0], h = sizes[s][1];
		const bool full = w == 4144;
		const size_t n = (size_t)w * h;
		uint16_t *mono16 = test_make_bayer16(w, h, 20 + s, 300);
		uint8_t *mono8 = test_make_bayer8(w, h, 20 + s, 300);
		uint16_t *rgb48 = test_make_rgb48(w, h, 20 + s, 300);
		uint8_t *rgb24 = malloc(n * 3);
		uint8_t *rgba32 = malloc(n * 4);
		uint8_t *abgr32 = malloc(n * 4);
		for (size_t i = 0; i < n; i++) {
			unsigned r = rgb48[3 * i] >> 4, g = rgb48[3 * i + 1] >> 4, b = rgb48[3 * i + 2] >> 4;
			r = r > 255 ? 255 : r; g = g > 255 ? 255 : g; b = b > 255 ? 255 : b;
			rgb24[3 * i] = r; rgb24[3 * i + 1] = g; rgb24[3 * i + 2] = b;
			rgba32[4 * i] = r; rgba32[4 * i + 1] = g; rgba32[4 * i + 2] = b; rgba32[4 * i + 3] = 255;
			abgr32[4 * i] = 255; abgr32[4 * i + 1] = b; abgr32[4 * i + 2] = g; abgr32[4 * i + 3] = r;
		}
		char label[64];
		for (double sigma = 1.0; sigma <= 3.0; sigma += 1.0) {
			snprintf(label, sizeof(label), "mono8 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_MONO8, mono8, w, h, sigma, full && sigma == 2.0);
			snprintf(label, sizeof(label), "mono16 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_MONO16, mono16, w, h, sigma, full && sigma == 2.0);
			snprintf(label, sizeof(label), "rgb24 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_RGB24, rgb24, w, h, sigma, false);
			snprintf(label, sizeof(label), "rgba32 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_RGBA32, rgba32, w, h, sigma, false);
			snprintf(label, sizeof(label), "abgr32 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_ABGR32, abgr32, w, h, sigma, false);
			snprintf(label, sizeof(label), "rgb48 sigma %.0f", sigma); check_binarize(label, INDIGO_RAW_RGB48, rgb48, w, h, sigma, false);
		}
		// constant and uniform-noise frames: MAD/stddev edge cases (0 and maximal)
		uint8_t *flat8 = malloc(n);
		uint16_t *flat16 = malloc(n * 2);
		memset(flat8, 77, n);
		for (size_t i = 0; i < n; i++) flat16[i] = 1234;
		check_binarize("constant 8 bit", INDIGO_RAW_MONO8, flat8, w, h, 2.0, false);
		check_binarize("constant 16 bit", INDIGO_RAW_MONO16, flat16, w, h, 2.0, false);
		free(flat8);
		free(flat16);

		check_equalize("mono16", INDIGO_RAW_MONO16, mono16, w, h, full);
		check_equalize("mono8", INDIGO_RAW_MONO8, mono8, w, h, full);

		free(mono16);
		free(mono8);
		free(rgb48);
		free(rgb24);
		free(rgba32);
		free(abgr32);
	}
	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
