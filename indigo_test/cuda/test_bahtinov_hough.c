// Verifies that the optimized hough_transform() (trig lookup table) and find_hough_max() (cache-friendly loop
// order, same explicit tie-break as the original) return exactly the same result as the original implementation
// (git HEAD before Phase 4), on realistic Bahtinov-mask-like skeletons and on synthetic accumulators built to
// contain deliberate ties (the case the loop reorder has to get right).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>

// white-box hooks into the (non-static, but not in any public header) internals of indigo_raw_utils.c
extern void indigo_make_trig_table_for_test(double *cos_table, double *sin_table);
extern void indigo_hough_transform_for_test(uint8_t *data, int width, int height, int *hough, const double *cos_table, const double *sin_table);
extern int indigo_find_hough_max_for_test(int *hough, int width, int height, double *rho, double *theta);

#define RHO_RES   3000
#define THETA_RES 3000

// ---- reference: original implementation (git HEAD before Phase 4), only renamed ----
static void reference_hough_transform(uint8_t* data, int width, int height, int *hough) {
	uint8_t (*pixels)[width] = (uint8_t (*)[width]) data;
	int (*acc)[THETA_RES] = (int (*)[THETA_RES]) hough;
	double theta_step = M_PI / THETA_RES;
	double rho_step = 2 * sqrt(width * width + height * height) / RHO_RES;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (pixels[y][x] > 0) {
				for (int theta_index = 0; theta_index < THETA_RES; theta_index++) {
					double theta = theta_index * theta_step;
					double rho = x * cos(theta) + y * sin(theta);
					int rho_index = (int)(rho / rho_step) + RHO_RES / 2;
					acc[rho_index][theta_index]++;
				}
			}
		}
	}
}

static int reference_find_hough_max(int *hough, int width, int height, double *rho, double *theta) {
	int (*acc)[THETA_RES] = (int (*)[THETA_RES]) hough;
	int max = 0;
	int max_rho_index = 0;
	int max_theta_index = 0;
	double theta_step = M_PI / THETA_RES;
	double rho_step = 2 * sqrt(width * width + height * height) / RHO_RES;
	for (int theta_index = 0; theta_index < THETA_RES; theta_index++) {
		for (int rho_index = 0; rho_index < RHO_RES; rho_index++) {
			int count = acc[rho_index][theta_index];
			if (count > max) {
				max = count;
				max_rho_index = rho_index;
				max_theta_index = theta_index;
			}
		}
	}
	int DIFF = M_PI / 180 / theta_step;
	for (int diff = -DIFF; diff < DIFF; diff++) {
		int theta_index = max_theta_index + diff;
		if (theta_index < 0) continue;
		if (theta_index >= THETA_RES) break;
		for (int rho_index = 0; rho_index < RHO_RES; rho_index++) {
			acc[rho_index][theta_index] = 0;
		}
	}
	*rho = (max_rho_index - RHO_RES / 2) * rho_step;
	*theta = max_theta_index * theta_step;
	return max;
}
// ---- end of reference ----

static int checks = 0, failures = 0;

static bool same_double(double a, double b) { return memcmp(&a, &b, sizeof(double)) == 0; }

// runs reference and library find_hough_max on the same accumulator (each on its own copy) for `lines` rounds
static void compare_find_hough_max(const char *label, const int *seed_hough, int width, int height, int lines) {
	size_t bytes = (size_t)RHO_RES * THETA_RES * sizeof(int);
	int *ref_acc = malloc(bytes), *lib_acc = malloc(bytes);
	memcpy(ref_acc, seed_hough, bytes);
	memcpy(lib_acc, seed_hough, bytes);
	for (int line = 0; line < lines; line++) {
		double ref_rho, ref_theta, lib_rho, lib_theta;
		int ref_max = reference_find_hough_max(ref_acc, width, height, &ref_rho, &ref_theta);
		int lib_max = indigo_find_hough_max_for_test(lib_acc, width, height, &lib_rho, &lib_theta);
		checks++;
		bool ok = ref_max == lib_max && same_double(ref_rho, lib_rho) && same_double(ref_theta, lib_theta);
		if (!ok) {
			failures++;
			printf("FAIL %s line %d: max %d/%d rho %.17g/%.17g theta %.17g/%.17g\n", label, line, ref_max, lib_max, ref_rho, lib_rho, ref_theta, lib_theta);
		}
	}
	free(ref_acc);
	free(lib_acc);
}

static void compare_hough_transform_and_search(const char *label, uint8_t *mask, int width, int height, int lines) {
	size_t bytes = (size_t)RHO_RES * THETA_RES * sizeof(int);
	int *ref_hough = calloc(1, bytes);
	int *lib_hough = calloc(1, bytes);
	double cos_table[THETA_RES], sin_table[THETA_RES];
	indigo_make_trig_table_for_test(cos_table, sin_table);
	reference_hough_transform(mask, width, height, ref_hough);
	indigo_hough_transform_for_test(mask, width, height, lib_hough, cos_table, sin_table);
	checks++;
	if (memcmp(ref_hough, lib_hough, bytes) != 0) {
		failures++;
		printf("FAIL %s: hough_transform accumulator differs\n", label);
	}
	// and the search on top of it (this is what indigo_bahtinov_error actually uses)
	compare_find_hough_max(label, ref_hough, width, height, lines);
	free(ref_hough);
	free(lib_hough);
}

// a few crossing lines, like Bahtinov diffraction spikes, plus some noise pixels
static uint8_t *make_mask(int width, int height, uint64_t seed, int line_count) {
	uint8_t *mask = calloc((size_t)width * height, 1);
	uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1;
	#define NEXT() (state ^= state >> 12, state ^= state << 25, state ^= state >> 27, (uint32_t)((state * 0x2545F4914F6CDD1DULL) >> 32))
	double cx = width / 2.0, cy = height / 2.0;
	for (int l = 0; l < line_count; l++) {
		double angle = (NEXT() % 18000) / 100.0 * M_PI / 180.0;
		double dx = cos(angle), dy = sin(angle);
		for (double t = -width; t < width; t += 0.5) {
			int x = (int)round(cx + t * dx), y = (int)round(cy + t * dy);
			if (x >= 0 && x < width && y >= 0 && y < height) mask[(size_t)y * width + x] = 255;
		}
	}
	for (int i = 0; i < width * height / 500; i++) {
		mask[NEXT() % ((size_t)width * height)] = 255;
	}
	return mask;
}

int main() {
	// realistic masks at a few sizes (including the 500x500 cap the imager agent enforces)
	const int sizes[][2] = { { 50, 50 }, { 200, 200 }, { 500, 500 }, { 500, 300 } };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		int w = sizes[s][0], h = sizes[s][1];
		uint8_t *mask = make_mask(w, h, 10 + s, 3);
		char label[32];
		snprintf(label, sizeof(label), "%dx%d lines", w, h);
		compare_hough_transform_and_search(label, mask, w, h, 15);
		free(mask);
	}

	// empty mask (no votes at all - every count is 0, DIFF suppression must not crash)
	{
		uint8_t *mask = calloc(200 * 200, 1);
		compare_hough_transform_and_search("empty 200x200", mask, 200, 200, 15);
		free(mask);
	}

	// full mask (every pixel set - heavy, but only run a couple of lines to keep the test fast)
	{
		uint8_t *mask = malloc(80 * 80);
		memset(mask, 255, 80 * 80);
		compare_hough_transform_and_search("full 80x80", mask, 80, 80, 3);
		free(mask);
	}

	// synthetic accumulators built to contain deliberate ties at the maximum count - this is the case the
	// loop reorder in find_hough_max has to get exactly right (original tie-break: lower theta_index wins,
	// then lower rho_index)
	{
		size_t bytes = (size_t)RHO_RES * THETA_RES * sizeof(int);
		int *hough = calloc(1, bytes);
		int (*acc)[THETA_RES] = (int (*)[THETA_RES]) hough;
		// three cells share the global max 7: two different theta_index (smallest must win), and within the
		// smallest theta_index, two different rho_index (smallest must win)
		acc[100][50] = 7;
		acc[50][50] = 7;
		acc[10][20] = 7;
		acc[2000][2999] = 6; // a distinct runner-up for the second find_hough_max round
		compare_find_hough_max("deliberate tie", hough, 500, 500, 3);
		free(hough);
	}
	{
		// tie spanning the DIFF suppression boundary: the winning theta_index of round 1 must be the one
		// actually returned, and the tie among rho_index within it must resolve to the smallest
		size_t bytes = (size_t)RHO_RES * THETA_RES * sizeof(int);
		int *hough = calloc(1, bytes);
		int (*acc)[THETA_RES] = (int (*)[THETA_RES]) hough;
		for (int t = 0; t < THETA_RES; t += 137) {
			acc[(t * 37) % RHO_RES][t] = 5; // scattered equal-count cells across many columns
		}
		acc[42][7] = 9;
		acc[42][1234] = 9; // ties the max in a later column - theta_index 7 must still win
		compare_find_hough_max("scattered tie", hough, 500, 500, 5);
		free(hough);
	}

	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
