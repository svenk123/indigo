// Synthetic but realistic astro frames for the CUDA tests and benchmarks:
// vignetting + gradient, shot/read noise, Gaussian stars (some saturated), hot pixels, RGGB Bayer pattern.
// Deterministic for a given seed.

#ifndef test_image_h
#define test_image_h

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	uint64_t state;
} test_rng;

static inline uint32_t test_rng_u32(test_rng *rng) {
	rng->state ^= rng->state >> 12;
	rng->state ^= rng->state << 25;
	rng->state ^= rng->state >> 27;
	return (uint32_t)((rng->state * 0x2545F4914F6CDD1DULL) >> 32);
}

static inline double test_rng_uniform(test_rng *rng) {
	return (test_rng_u32(rng) + 0.5) / 4294967296.0;
}

static inline double test_rng_gauss(test_rng *rng) {
	return sqrt(-2 * log(test_rng_uniform(rng))) * cos(2 * M_PI * test_rng_uniform(rng));
}

// RGGB mono16 frame, values 0..65535
static inline uint16_t *test_make_bayer16(int width, int height, uint64_t seed, int star_count) {
	test_rng rng = { seed * 0x9E3779B97F4A7C15ULL + 1 };
	const double gain[4] = { 0.9, 1.0, 1.0, 0.75 }; // R G G B
	float *frame = (float *)malloc((size_t)width * height * sizeof(float));
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double dx = (x - width / 2.0) / (width / 2.0), dy = (y - height / 2.0) / (height / 2.0);
			double background = 900 * (1 - 0.2 * (dx * dx + dy * dy)) * (1 + 0.05 * dx);
			frame[(size_t)y * width + x] = (float)(background * gain[(y & 1) * 2 + (x & 1)]);
		}
	}
	for (int i = 0; i < star_count; i++) {
		double x0 = test_rng_uniform(&rng) * width, y0 = test_rng_uniform(&rng) * height;
		double flux = 2000 * pow(1000.0, test_rng_uniform(&rng)); // 2e3 .. 2e6
		double sigma = (2.5 + 2 * test_rng_uniform(&rng)) / 2.355;
		double amplitude = flux / (2 * M_PI * sigma * sigma);
		int radius = (int)(5 * sigma) + 1;
		for (int y = (int)y0 - radius; y <= (int)y0 + radius; y++) {
			for (int x = (int)x0 - radius; x <= (int)x0 + radius; x++) {
				if (x < 0 || y < 0 || x >= width || y >= height) {
					continue;
				}
				double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
				frame[(size_t)y * width + x] += (float)(amplitude * exp(-r2 / (2 * sigma * sigma)) * gain[(y & 1) * 2 + (x & 1)]);
			}
		}
	}
	uint16_t *result = (uint16_t *)malloc((size_t)width * height * sizeof(uint16_t));
	for (size_t i = 0; i < (size_t)width * height; i++) {
		double value = frame[i] + test_rng_gauss(&rng) * sqrt(frame[i] * 0.05 + 30);
		result[i] = value < 0 ? 0 : value > 65535 ? 65535 : (uint16_t)value;
	}
	for (int i = 0; i < width * height / 60000; i++) { // hot pixels
		result[(size_t)(test_rng_uniform(&rng) * width * height)] = 40000 + (uint16_t)(test_rng_uniform(&rng) * 20000);
	}
	free(frame);
	return result;
}

// 8 bit version of the same frame (scaled so that background and stars use a useful part of the range)
static inline uint8_t *test_make_bayer8(int width, int height, uint64_t seed, int star_count) {
	uint16_t *frame16 = test_make_bayer16(width, height, seed, star_count);
	uint8_t *result = (uint8_t *)malloc((size_t)width * height);
	for (size_t i = 0; i < (size_t)width * height; i++) {
		unsigned value = frame16[i] >> 4;
		result[i] = value > 255 ? 255 : (uint8_t)value;
	}
	free(frame16);
	return result;
}

// Interleaved RGB (3 components per pixel) from a Bayer frame - crude nearest neighbour, good enough for statistics
static inline uint16_t *test_make_rgb48(int width, int height, uint64_t seed, int star_count) {
	uint16_t *bayer = test_make_bayer16(width, height, seed, star_count);
	uint16_t *result = (uint16_t *)malloc((size_t)width * height * 3 * sizeof(uint16_t));
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int x0 = x & ~1, y0 = y & ~1;
			if (x0 + 1 >= width) x0 = width - 2;
			if (y0 + 1 >= height) y0 = height - 2;
			if (x0 < 0) x0 = 0;
			if (y0 < 0) y0 = 0;
			size_t o = ((size_t)y * width + x) * 3;
			result[o] = bayer[(size_t)y0 * width + x0];
			result[o + 1] = bayer[(size_t)y0 * width + (x0 + 1 < width ? x0 + 1 : x0)];
			result[o + 2] = bayer[(size_t)(y0 + 1 < height ? y0 + 1 : y0) * width + (x0 + 1 < width ? x0 + 1 : x0)];
		}
	}
	free(bayer);
	return result;
}

#endif
