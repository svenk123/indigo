// Compares the stretch / debayer functions of the preview on the GPU with the CPU code (switched with
// indigo_cuda_set_enabled): the output must be byte-identical. Frames below INDIGO_CUDA_MIN_PIXELS stay on the CPU
// (checked as well). Also prints the timing. Without CUDA (make CUDA=no) the test is skipped.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_stretch.h>

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

typedef struct {
	int bits; // 8, 16, 24, 48
	int pattern; // 0 rggb, 1 gbrg, 2 grbg, 3 bggr; -1 - not bayer
	bool debayer_only;
	const void *data;
	int width, height;
} job;

static const char *pattern_names[] = { "rggb", "gbrg", "grbg", "bggr" };

// runs one function of the library (stretch parameters are always computed on the CPU)
static void run_job(const job *j, uint8_t *out, size_t out_size) {
	double sh[3], mi[3], hi[3];
	unsigned long *hist[3] = { NULL, NULL, NULL }, tot[3];
	const int w = j->width, h = j->height, sample_by = w < 0x1FF ? 1 : w / 0x1FF;
	memset(out, 0xA5, out_size);
	if (j->debayer_only) {
		switch (j->pattern) {
			case 0: indigo_debayer_8_rggb(j->data, w, h, out); break;
			case 1: indigo_debayer_8_gbrg(j->data, w, h, out); break;
			case 2: indigo_debayer_8_grbg(j->data, w, h, out); break;
			default: indigo_debayer_8_bggr(j->data, w, h, out); break;
		}
		return;
	}
	#define BAYER(bits, fn) indigo_compute_stretch_params_##bits##_##fn(j->data, w, h, sample_by, sh, mi, hi, hist, tot, 0.25, -2.8); indigo_stretch_##bits##_##fn(j->data, w, h, out, sh, mi, hi, tot);
	if (j->bits == 16 && j->pattern >= 0) {
		switch (j->pattern) { case 0: BAYER(16, rggb) break; case 1: BAYER(16, gbrg) break; case 2: BAYER(16, grbg) break; default: BAYER(16, bggr) break; }
	} else if (j->bits == 8 && j->pattern >= 0) {
		switch (j->pattern) { case 0: BAYER(8, rggb) break; case 1: BAYER(8, gbrg) break; case 2: BAYER(8, grbg) break; default: BAYER(8, bggr) break; }
	} else if (j->bits == 16) {
		indigo_compute_stretch_params_16(j->data, w, h, sample_by, sh, mi, hi, hist, 0.25, -2.8);
		indigo_stretch_16(j->data, w, h, out, sh, mi, hi);
	} else if (j->bits == 8) {
		indigo_compute_stretch_params_8(j->data, w, h, sample_by, sh, mi, hi, hist, 0.25, -2.8);
		indigo_stretch_8(j->data, w, h, out, sh, mi, hi);
	} else if (j->bits == 24) {
		indigo_compute_stretch_params_24(j->data, w, h, sample_by, sh, mi, hi, hist, tot, 0.25, -2.8);
		indigo_stretch_24(j->data, w, h, out, sh, mi, hi, tot);
	} else {
		indigo_compute_stretch_params_48(j->data, w, h, sample_by, sh, mi, hi, hist, tot, 0.25, -2.8);
		indigo_stretch_48(j->data, w, h, out, sh, mi, hi, tot);
	}
	for (int i = 0; i < 3; i++) free(hist[i]);
}

static void compare(const char *label, const job *j, bool timing) {
	const size_t size = (size_t)j->width * j->height;
	const size_t out_size = size * ((j->bits == 8 || j->bits == 16) && j->pattern < 0 ? 1 : 3);
	uint8_t *cpu = malloc(out_size), *gpu = malloc(out_size);
	const bool gpu_expected = size >= INDIGO_CUDA_MIN_PIXELS;
	indigo_cuda_set_enabled(false);
	run_job(j, cpu, out_size);
	indigo_cuda_set_enabled(true);
	const unsigned long before = indigo_cuda_stretch_calls();
	run_job(j, gpu, out_size);
	const bool on_gpu = indigo_cuda_stretch_calls() > before;
	checks++;
	size_t different = 0, first = 0;
	int max_diff = 0;
	for (size_t i = 0; i < out_size; i++) {
		if (cpu[i] != gpu[i]) {
			if (different++ == 0) first = i;
			int d = abs((int)cpu[i] - (int)gpu[i]);
			if (d > max_diff) max_diff = d;
		}
	}
	bool ok = different == 0 && on_gpu == gpu_expected;
	if (!ok) failures++;
	printf("%s %-26s %5dx%-5d %s%s", ok ? "ok  " : "FAIL", label, j->width, j->height, on_gpu ? "GPU" : "CPU", gpu_expected == on_gpu ? "" : " (unexpected)");
	if (different) printf(", %zu of %zu bytes differ (max %d, first at %zu: cpu %d gpu %d)", different, out_size, max_diff, first, cpu[first], gpu[first]);
	if (timing) {
		double t_cpu[5], t_gpu[5];
		for (int r = 0; r < 5; r++) {
			indigo_cuda_set_enabled(false);
			double s = now_ms(); run_job(j, cpu, out_size); t_cpu[r] = now_ms() - s;
			indigo_cuda_set_enabled(true);
			s = now_ms(); run_job(j, gpu, out_size); t_gpu[r] = now_ms() - s;
		}
		qsort(t_cpu, 5, sizeof(double), compare_double);
		qsort(t_gpu, 5, sizeof(double), compare_double);
		printf(", CPU %.0f ms, GPU %.0f ms (incl. parameters on the CPU)", t_cpu[2], t_gpu[2]);
	}
	printf("\n");
	free(cpu);
	free(gpu);
}

int main() {
	if (!indigo_cuda_available()) {
		printf("SKIPPED: no CUDA device or built without CUDA\n");
		return 0;
	}
	printf("device: %s\n", indigo_cuda_device_name());
	const int sizes[][2] = { { 301, 203 }, { 640, 481 }, { 1001, 803 }, { 2049, 1027 }, { 4144, 2822 } };
	static const int off[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		const int w = sizes[s][0], h = sizes[s][1];
		const bool full = w == 4144;
		const size_t n = (size_t)w * h;
		uint16_t *big16 = test_make_bayer16(w + 2, h + 2, 3 + s, 300);
		uint8_t *big8 = test_make_bayer8(w + 2, h + 2, 3 + s, 300);
		for (int p = 0; p < (full ? 1 : 4); p++) {
			uint16_t *d16 = malloc(n * 2);
			uint8_t *d8 = malloc(n);
			for (int y = 0; y < h; y++) {
				memcpy(d16 + (size_t)y * w, big16 + (size_t)(y + off[p][1]) * (w + 2) + off[p][0], w * 2);
				memcpy(d8 + (size_t)y * w, big8 + (size_t)(y + off[p][1]) * (w + 2) + off[p][0], w);
			}
			char label[64];
			job j16 = { 16, p, false, d16, w, h }, j8 = { 8, p, false, d8, w, h }, jd = { 8, p, true, d8, w, h };
			snprintf(label, sizeof(label), "16 bit %s stretch", pattern_names[p]); compare(label, &j16, full);
			snprintf(label, sizeof(label), "8 bit %s stretch", pattern_names[p]); compare(label, &j8, full);
			snprintf(label, sizeof(label), "8 bit %s debayer only", pattern_names[p]); compare(label, &jd, full);
			if (p == 0) {
				job m16 = { 16, -1, false, d16, w, h }, m8 = { 8, -1, false, d8, w, h };
				compare("16 bit mono", &m16, full);
				compare("8 bit mono", &m8, full);
			}
			free(d16);
			free(d8);
		}
		uint16_t *rgb48 = test_make_rgb48(w, h, 3 + s, 300);
		uint8_t *rgb24 = malloc(n * 3);
		for (size_t i = 0; i < n * 3; i++) { unsigned v = rgb48[i] >> 4; rgb24[i] = v > 255 ? 255 : v; }
		job r24 = { 24, -1, false, rgb24, w, h }, r48 = { 48, -1, false, rgb48, w, h };
		compare("24 bit RGB", &r24, full);
		compare("48 bit RGB", &r48, full);
		free(rgb48);
		free(rgb24);
		free(big16);
		free(big8);
	}
	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
