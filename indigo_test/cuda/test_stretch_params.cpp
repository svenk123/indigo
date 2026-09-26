// Verifies that the histogram based indigo_compute_stretch_params_*() returns bit-identical results
// to the original implementation (sorting with std::nth_element), which is kept below as reference.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <time.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_stretch.h>

#include "test_image.h"

// ---- reference: original implementation (git 5f846e79f), only renamed ----
template <typename T> void reference_compute_stretch_params(const T *buffer, int width, int height, int sample_columns_by, int sample_rows_by, double *shadows, double *midtones, double *highlights, unsigned long *histogram, unsigned long *totals, float B = 0.25, float C = -2.8) {
	const int sample_size = ceil((float)width / sample_columns_by) * ceil((float)height / sample_rows_by);
	const int sample_size_2 = sample_size / 2;
	const int histo_divider = (sizeof(T) == 1) ? 1 : 256; // TBD for 32 bits
	unsigned long total = 0;
	std::vector<T> samples(sample_size);
	if (sample_rows_by == 1) {
		int i = 0;
		int size = width * height;
		for (int index = 0; index < size; index += sample_columns_by) {
			T value = buffer[index];
			histogram[(samples[i++] = value) / histo_divider]++;
			total += value;
		}
	} else {
		int i = 0;
		const T *line = buffer;
		for (int line_index = 0; line_index < height; line_index += sample_rows_by) {
			for (int column_index = 0; column_index < width; column_index += sample_columns_by) {
				T value = line[column_index];
				histogram[(samples[i++] = value) / histo_divider]++;
				total += value;
			}
			line += width * sample_rows_by;
		}
	}
	if (totals) {
		*totals = total;
	}
	std::nth_element(samples.begin(), samples.begin() + sample_size_2, samples.end());
	const float median_sample = samples[sample_size_2];
	// Find the Median deviation: 1.4826 * median of abs(sample[i] - median).
	std::vector<T> deviations(sample_size);
	for (int i = 0; i < sample_size; i++) {
		deviations[i] = abs(median_sample - samples[i]);
	}
	std::nth_element(deviations.begin(), deviations.begin() + sample_size / 2, deviations.end());
	// scale to 0 -> 1.0.
	const float input_range = (sizeof(T) == 1) ? 0xFFL : 0XFFFFL; // TBD for 32 bits
	const float median_deviation = deviations[sample_size_2];
	const float normalized_median = median_sample / input_range;
	const float MADN = 1.4826 * median_deviation / input_range;
	const bool upperHalf = normalized_median > 0.5;
	*shadows = (upperHalf || MADN == 0) ? 0.0 : fmin(1.0, fmax(0.0, (normalized_median + C * MADN)));
	*highlights = (!upperHalf || MADN == 0) ? 1.0 : fmin(1.0, fmax(0.0, (normalized_median - C * MADN)));
	float X, M;
	if (upperHalf) {
		X = B;
		M = *highlights - normalized_median;
	} else {
		X = normalized_median - *shadows;
		M = B;
	}
	if (X == 0) {
		*midtones = 0.0f;
	} else if (X == M) {
		*midtones = 0.5f;
	} else if (X == 1) {
		*midtones = 1.0f;
	} else {
		*midtones = ((M - 1) * X) / ((2 * M - 1) * X - M);
	}
}
// ---- end of reference ----

static int checks = 0, failures = 0;

struct result {
	double shadows[3], midtones[3], highlights[3];
	unsigned long histogram[3][256], totals[3];
	int channels;
};

static bool same(const char *name, int width, int height, const result &expected, const result &actual, float B, float C) {
	checks++;
	for (int i = 0; i < expected.channels; i++) {
		bool ok = memcmp(&expected.shadows[i], &actual.shadows[i], sizeof(double)) == 0 && memcmp(&expected.midtones[i], &actual.midtones[i], sizeof(double)) == 0 && memcmp(&expected.highlights[i], &actual.highlights[i], sizeof(double)) == 0 && memcmp(expected.histogram[i], actual.histogram[i], sizeof(expected.histogram[i])) == 0 && expected.totals[i] == actual.totals[i];
		if (!ok) {
			failures++;
			printf("FAIL %s %dx%d B=%g C=%g channel %d: shadows %.17g/%.17g midtones %.17g/%.17g highlights %.17g/%.17g totals %lu/%lu\n", name, width, height, B, C, i, expected.shadows[i], actual.shadows[i], expected.midtones[i], actual.midtones[i], expected.highlights[i], actual.highlights[i], expected.totals[i], actual.totals[i]);
			return false;
		}
	}
	return true;
}

// offsets of the R, G, B sample planes of a Bayer pattern
static void bayer_offsets(int pattern, int width, int *o) {
	// rggb: R=0 G=1 B=w+1, gbrg: R=w G=0 B=1, grbg: R=1 G=0 B=w, bggr: R=w+1 G=1 B=0
	switch (pattern) {
		case 0: o[0] = 0; o[1] = 1; o[2] = width + 1; break;
		case 1: o[0] = width; o[1] = 0; o[2] = 1; break;
		case 2: o[0] = 1; o[1] = 0; o[2] = width; break;
		default: o[0] = width + 1; o[1] = 1; o[2] = 0; break;
	}
}

template <typename T> static void run_bayer(const char *name, const T *data, int width, int height, int pattern, float B, float C, void (*lib)(const T *, int, int, int, double *, double *, double *, unsigned long **, unsigned long *, float, float)) {
	const int sample_by = width < 0x1FF ? 1 : width / 0x1FF;
	result expected, actual;
	memset(&expected, 0, sizeof(expected));
	memset(&actual, 0, sizeof(actual));
	expected.channels = actual.channels = 3;
	int o[3];
	bayer_offsets(pattern, width, o);
	for (int i = 0; i < 3; i++) {
		reference_compute_stretch_params(data + o[i], width, height, sample_by * 2, 2, &expected.shadows[i], &expected.midtones[i], &expected.highlights[i], expected.histogram[i], &expected.totals[i], B, C);
	}
	unsigned long *histogram[3];
	lib(data, width, height, sample_by, actual.shadows, actual.midtones, actual.highlights, histogram, actual.totals, B, C);
	for (int i = 0; i < 3; i++) {
		memcpy(actual.histogram[i], histogram[i], sizeof(actual.histogram[i]));
		free(histogram[i]);
	}
	same(name, width, height, expected, actual, B, C);
}

// interleaved RGB: 24 (8 bit) / 48 (16 bit)
template <typename T> static void run_rgb(const char *name, const T *data, int width, int height, float B, float C, void (*lib)(const T *, int, int, int, double *, double *, double *, unsigned long **, unsigned long *, float, float)) {
	const int sample_by = width < 0x1FF ? 1 : width / 0x1FF;
	result expected, actual;
	memset(&expected, 0, sizeof(expected));
	memset(&actual, 0, sizeof(actual));
	expected.channels = actual.channels = 3;
	for (int i = 0; i < 3; i++) {
		reference_compute_stretch_params(data + i, 3 * width, height, sample_by * 3, 1, &expected.shadows[i], &expected.midtones[i], &expected.highlights[i], expected.histogram[i], &expected.totals[i], B, C);
	}
	unsigned long *histogram[3];
	lib(data, width, height, sample_by, actual.shadows, actual.midtones, actual.highlights, histogram, actual.totals, B, C);
	for (int i = 0; i < 3; i++) {
		memcpy(actual.histogram[i], histogram[i], sizeof(actual.histogram[i]));
		free(histogram[i]);
	}
	same(name, width, height, expected, actual, B, C);
}

// mono: indigo_compute_stretch_params_8 / _16 (no totals)
template <typename T> static void run_mono(const char *name, const T *data, int width, int height, float B, float C, void (*lib)(const T *, int, int, int, double *, double *, double *, unsigned long **, float, float)) {
	const int sample_by = width < 0x1FF ? 1 : width / 0x1FF;
	result expected, actual;
	memset(&expected, 0, sizeof(expected));
	memset(&actual, 0, sizeof(actual));
	expected.channels = actual.channels = 1;
	reference_compute_stretch_params(data, width, height, sample_by, 1, &expected.shadows[0], &expected.midtones[0], &expected.highlights[0], expected.histogram[0], (unsigned long *)NULL, B, C);
	unsigned long *histogram[3];
	lib(data, width, height, sample_by, actual.shadows, actual.midtones, actual.highlights, histogram, B, C);
	memcpy(actual.histogram[0], histogram[0], sizeof(actual.histogram[0]));
	free(histogram[0]);
	same(name, width, height, expected, actual, B, C);
}

// The sampling of the Bayer planes that start one row down reads one row behind the buffer if the height is odd
// (pre-existing behaviour of the original code). The buffers are zero padded here so that both implementations
// read the same values.
static void run_all(const char *label, const uint16_t *frame16, const uint8_t *frame8, int width, int height) {
	const size_t size = (size_t)width * height, padding = 2 * (size_t)width + 16;
	uint16_t *data16 = (uint16_t *)calloc(size + padding, sizeof(uint16_t));
	uint8_t *data8 = (uint8_t *)calloc(size + padding, sizeof(uint8_t));
	memcpy(data16, frame16, size * sizeof(uint16_t));
	memcpy(data8, frame8, size * sizeof(uint8_t));
	const float params[2][2] = { { 0.25f, -2.8f }, { 0.1f, -1.5f } };
	for (int p = 0; p < 2; p++) {
		float B = params[p][0], C = params[p][1];
		char name[128];
		snprintf(name, sizeof(name), "%s 16 rggb", label); run_bayer<uint16_t>(name, data16, width, height, 0, B, C, indigo_compute_stretch_params_16_rggb);
		snprintf(name, sizeof(name), "%s 16 gbrg", label); run_bayer<uint16_t>(name, data16, width, height, 1, B, C, indigo_compute_stretch_params_16_gbrg);
		snprintf(name, sizeof(name), "%s 16 grbg", label); run_bayer<uint16_t>(name, data16, width, height, 2, B, C, indigo_compute_stretch_params_16_grbg);
		snprintf(name, sizeof(name), "%s 16 bggr", label); run_bayer<uint16_t>(name, data16, width, height, 3, B, C, indigo_compute_stretch_params_16_bggr);
		snprintf(name, sizeof(name), "%s 8 rggb", label); run_bayer<uint8_t>(name, data8, width, height, 0, B, C, indigo_compute_stretch_params_8_rggb);
		snprintf(name, sizeof(name), "%s 8 gbrg", label); run_bayer<uint8_t>(name, data8, width, height, 1, B, C, indigo_compute_stretch_params_8_gbrg);
		snprintf(name, sizeof(name), "%s 8 grbg", label); run_bayer<uint8_t>(name, data8, width, height, 2, B, C, indigo_compute_stretch_params_8_grbg);
		snprintf(name, sizeof(name), "%s 8 bggr", label); run_bayer<uint8_t>(name, data8, width, height, 3, B, C, indigo_compute_stretch_params_8_bggr);
		snprintf(name, sizeof(name), "%s 16 mono", label); run_mono<uint16_t>(name, data16, width, height, B, C, indigo_compute_stretch_params_16);
		snprintf(name, sizeof(name), "%s 8 mono", label); run_mono<uint8_t>(name, data8, width, height, B, C, indigo_compute_stretch_params_8);
		snprintf(name, sizeof(name), "%s 24 rgb", label); run_rgb<uint8_t>(name, data8, width / 3 > 0 ? width / 3 : 1, height, B, C, indigo_compute_stretch_params_24);
		snprintf(name, sizeof(name), "%s 48 rgb", label); run_rgb<uint16_t>(name, data16, width / 3 > 0 ? width / 3 : 1, height, B, C, indigo_compute_stretch_params_48);
	}
	free(data16);
	free(data8);
}

int main() {
	// sizes: tiny, odd, below/above the sampling threshold (0x1FF), multiples and non multiples of the sample step, full frame
	const int sizes[][2] = { { 4, 4 }, { 7, 5 }, { 64, 48 }, { 511, 301 }, { 512, 512 }, { 1023, 767 }, { 1600, 1200 }, { 2048, 1536 }, { 3001, 999 }, { 4144, 2822 } };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		int width = sizes[s][0], height = sizes[s][1];
		size_t size = (size_t)width * height;
		char label[64];
		snprintf(label, sizeof(label), "astro");
		uint16_t *data16 = test_make_bayer16(width, height, 100 + s, size / 20000 + 3);
		uint8_t *data8 = test_make_bayer8(width, height, 100 + s, size / 20000 + 3);
		run_all(label, data16, data8, width, height);
		// pathological data: constant (MAD = 0), saturated, uniform noise over the full range, dark
		for (int kind = 0; kind < 4; kind++) {
			test_rng rng = { (uint64_t)(12345 + kind) };
			for (size_t i = 0; i < size; i++) {
				switch (kind) {
					case 0: data16[i] = 1234; data8[i] = 77; break;
					case 1: data16[i] = 65535; data8[i] = 255; break;
					case 2: data16[i] = test_rng_u32(&rng) & 0xFFFF; data8[i] = test_rng_u32(&rng) & 0xFF; break;
					default: data16[i] = test_rng_u32(&rng) % 3; data8[i] = test_rng_u32(&rng) % 3; break;
				}
			}
			static const char *kinds[] = { "constant", "saturated", "uniform", "dark" };
			run_all(kinds[kind], data16, data8, width, height);
		}
		free(data16);
		free(data8);
	}
	// timing on a full frame: original (sorting) against histogram implementation, 16 bit RGGB, three planes
	{
		const int width = 4144, height = 2822, sample_by = width / 0x1FF;
		const size_t size = (size_t)width * height;
		uint16_t *frame = (uint16_t *)calloc(size + 2 * width + 16, sizeof(uint16_t));
		uint16_t *astro = test_make_bayer16(width, height, 1, 600);
		memcpy(frame, astro, size * sizeof(uint16_t));
		free(astro);
		double reference_ms[5], library_ms[5];
		for (int run = 0; run < 5; run++) {
			struct timespec t0, t1, t2;
			result r;
			memset(&r, 0, sizeof(r));
			clock_gettime(CLOCK_MONOTONIC, &t0);
			const int o[3] = { 0, 1, width + 1 };
			for (int i = 0; i < 3; i++) {
				reference_compute_stretch_params(frame + o[i], width, height, sample_by * 2, 2, &r.shadows[i], &r.midtones[i], &r.highlights[i], r.histogram[i], &r.totals[i], 0.25f, -2.8f);
			}
			clock_gettime(CLOCK_MONOTONIC, &t1);
			unsigned long *histogram[3];
			indigo_compute_stretch_params_16_rggb(frame, width, height, sample_by, r.shadows, r.midtones, r.highlights, histogram, r.totals, 0.25f, -2.8f);
			clock_gettime(CLOCK_MONOTONIC, &t2);
			for (int i = 0; i < 3; i++) free(histogram[i]);
			reference_ms[run] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
			library_ms[run] = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6;
		}
		std::sort(reference_ms, reference_ms + 5);
		std::sort(library_ms, library_ms + 5);
		printf("timing 4144x2822 16 bit rggb, median of 5: original %.1f ms, histogram %.1f ms\n", reference_ms[2], library_ms[2]);
		free(frame);
	}
	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
