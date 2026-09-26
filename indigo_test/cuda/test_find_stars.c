// Verifies that indigo_find_stars_precise() (single pass over a sorted list of star peaks) returns exactly
// the same stars as the original implementation (full frame scan for every star), which is kept below as reference.
// Also prints the timing of both.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <sys/param.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>

#include "test_image.h"

// ---- reference: original implementation (git 528a18c50), only renamed ----
static int median3(int a, int b, int c) {
	if (a > b) {
		if (b > c) return b;
		else if (a > c) return c;
		else return a;
	} else {
		if (a > c) return a;
		else if (b > c) return c;
		else return b;
	}
}

static const double FIND_STAR_EDGE_CLIPPING = 20;

static int luminance_comparator(const void *item_1, const void *item_2) {
	if (((indigo_star_detection *)item_1)->luminance < ((indigo_star_detection *)item_2)->luminance)
		return 1;
	if (((indigo_star_detection *)item_1)->luminance > ((indigo_star_detection *)item_2)->luminance)
		return -1;
	return 0;
}

/* With radius < 3, no precise star positins will be determined */
indigo_result reference_find_stars_precise(indigo_raw_type raw_type, const void *data, const uint16_t radius, const int width, const int height, const int stars_max, indigo_star_detection star_list[], int *stars_found) {
	if (data == NULL || star_list == NULL || stars_found == NULL) return INDIGO_FAILED;
	
	int  size = width * height;
	uint16_t *buf = indigo_safe_malloc(size * sizeof(uint16_t));
	int star_size = 100;
	const int clip_edge = height >= FIND_STAR_EDGE_CLIPPING * 4 ? FIND_STAR_EDGE_CLIPPING : (height / 4);
	int clip_width  = width - clip_edge;
	int clip_height = height - clip_edge;
	uint16_t max_luminance = 0;
	
	uint8_t *data8 = (uint8_t *)data;
	uint16_t *data16 = (uint16_t *)data;
	double sum = 0;
	double sum_sq = 0;
	
	switch (raw_type) {
		case INDIGO_RAW_MONO8: {
			max_luminance = 0xFF;
			for (int i = 0; i < size; i++) {
				buf[i] = data8[i];
				sum += buf[i];
				sum_sq += buf[i] * buf[i];
			}
			break;
		}
		case INDIGO_RAW_MONO16: {
			max_luminance = 0xFFFF;
			for (int i = 0; i < size; i++) {
				buf[i] = data16[i];
				sum += buf[i];
				sum_sq += buf[i] * buf[i];
			}
			break;
		}
		case INDIGO_RAW_RGB24: {
			max_luminance = 0xFF;
			for (int i = 0, j = 0; i < 3 * size; i++, j++) {
				buf[j] = (data8[i] + data8[i + 1] + data8[i + 2]) / 3;
				sum += buf[j];
				sum_sq += buf[j] * buf[j];
				i += 2;
			}
			break;
		}
		case INDIGO_RAW_RGBA32: {
			max_luminance = 0xFF;
			for (int i = 0, j = 0; i < 4 * size; i++, j++) {
				buf[j] = (data8[i] + data8[i + 1] + data8[i + 2]) / 3;
				sum += buf[j];
				sum_sq += buf[j] * buf[j];
				i += 3;
			}
			break;
		}
		case INDIGO_RAW_ABGR32: {
			max_luminance = 0xFF;
			for (int i = 0, j = 0; i < 4 * size; i++, j++) {
				buf[j] = (data8[i + 1] + data8[i + 2] + data8[i + 3]) / 3;
				sum += buf[j];
				sum_sq += buf[j] * buf[j];
				i += 3;
			}
			break;
		}
		case INDIGO_RAW_RGB48: {
			max_luminance = 0xFFFF;
			for (int i = 0, j = 0; i < 3 * size; i++, j++) {
				buf[j] = (data16[i] + data16[i + 1] + data16[i + 2]) / 3;
				sum += buf[j];
				sum_sq += buf[j] * buf[j];
				i += 2;
			}
			break;
		}
	}
	
	// Calculate mean
	double mean = sum / size;
	
	/* Calculate standard deviation - simplified, approximate estimate,
	 with a nice property that it is less affected by outliers. This proeprty
	 fixes the issue with finding guide stars in the presence of saturated stars,
	 as it effectively filters out the outliers.
	 */
	double stddev = sqrt(fabs(sum_sq / size - mean * mean));
	
	/* Calculate threshold - add 4.5 stddev threshold for stars */
	uint32_t threshold = 4.5 * stddev + mean;
	indigo_debug("%s(): image mean = %.2f, simplified stddev = %.2f, star detection threshold = %d", __FUNCTION__, mean, stddev, threshold);
	
	int threshold_hist = threshold * 0.9;
	
	int found = 0;
	int width2 = width / 2;
	int height2 = height / 2;
	uint32_t lmax = threshold + 1;
	
	indigo_star_detection star = { 0 };
	int divider = (width > height) ? height2 : width2;
	while (lmax > threshold) {
		lmax = threshold;
		star.x = 0;
		star.y = 0;
		star.nc_distance = 0;
		star.luminance = 0;
		star.oversaturated = 0;
		
		for (int j = clip_edge; j < clip_height; j++) {
			for (int i = clip_edge; i < clip_width; i++) {
				int off = j * width + i;
				if (
						buf[off] > lmax &&
						/* also check median of the neighbouring pixels to avoid hot pixels and lines */
						median3(buf[off - 1], buf[off], buf[off + 1]) > threshold &&
						median3(buf[off - width], buf[off], buf[off + width]) > threshold &&
						median3(buf[off - width - 1], buf[off], buf[off + width + 1]) > threshold &&
						median3(buf[off - width + 1], buf[off], buf[off + width - 1]) > threshold
						) {
							lmax = buf[off];
							star.x = i;
							star.y = j;
						}
			}
		}
		if (lmax > threshold) {
			double luminance = 0;
			int min_i = MAX(0, star.x - star_size);
			int max_i = MIN(width - 1, star.x + star_size);
			int min_j = MAX(0, star.y - star_size);
			int max_j = MIN(height - 1, star.y + star_size);
			int star_x = (int)star.x;
			int star_y = (int)star.y;
			// clear +X, +Y quadrant
			for (int j = star_y; j <= max_j; j++) {
				if (buf[j * width + star_x] < threshold_hist) {
					break;
				}
				for (int i = star_x; i <= max_i; i++) {
					int off = j * width + i;
					if (buf[off] > threshold_hist) {
						luminance += buf[off] - threshold_hist;
						buf[off] = 0;
					} else {
						break;
					}
				}
			}
			// clear -X, +Y quadrant
			for (int j = star_y; j <= max_j; j++) {
				if (buf[j * width + star_x - 1] < threshold_hist) {
					break;
				}
				for (int i = star_x - 1; i >= min_i; i--) {
					int off = j * width + i;
					if (buf[off] > threshold_hist) {
						luminance += buf[off] - threshold_hist;
						buf[off] = 0;
					} else {
						break;
					}
				}
			}
			// clear +X, -Y quadrant
			for (int j = star_y - 1; j >= min_j; j--) {
				if (buf[j * width + star_x] < threshold_hist) {
					break;
				}
				for (int i = star_x; i <= max_i; i++) {
					int off = j * width + i;
					if (buf[off] > threshold_hist) {
						luminance += buf[off] - threshold_hist;
						buf[off] = 0;
					} else {
						break;
					}
				}
			}
			// clear -X, -Y quadrant
			for (int j = star_y - 1; j >= min_j; j--) {
				if (buf[j * width + star_x - 1] < threshold_hist) {
					break;
				}
				for (int i = star_x - 1; i >= min_i; i--) {
					int off = j * width + i;
					if (buf[off] > threshold_hist) {
						luminance += buf[off] - threshold_hist;
						buf[off] = 0;
					} else {
						break;
					}
				}
			}
			
			indigo_result res = INDIGO_FAILED;
			if (radius >= 3) {
				indigo_frame_digest center = {0};
				res = indigo_selection_frame_digest_iterative(raw_type, data, &star.x, &star.y, radius, width, height, &center, 2);
				star.x = center.centroid_x;
				star.y = center.centroid_y;
				star.close_to_other = false;
				if (res == INDIGO_OK) {
					indigo_delete_frame_digest(&center);
				}
			}
			
			/* Check if the star is a duplicate (probably artifact) or is in close proximity to another one.
			 In both cses these stars should not be used */
			if (res == INDIGO_OK || radius < 3) {
				for (int i = 0; i < found; i++) {
					double dx = fabs(star_list[i].x - star.x);
					double dy = fabs(star_list[i].y - star.y);
					if (dx < 1 && dy < 1) {
						/* The star (probably artifact) is a duplicate of another star.
						 We mark the other star as being close to another one, so it
						 won't be used automatically, and we skip the duplicate. */
						indigo_debug("indigo_find_stars(): star (%lf, %lf) skipped, duplicate of #%u = (%lf, %lf)", star.x, star.y, i + 1, star_list[i].x, star_list[i].y);
						star_list[i].close_to_other = true;
						res = INDIGO_FAILED;
						break;
					} else if (dx < radius && dy < radius) {
						/* The star is too close to another star.
						 We mark both star as being close to another one, so they
						 won't be used automatically but we keep both stars in the list. */
						indigo_debug("indigo_find_stars(): star (%lf, %lf), too close to #%u = (%lf, %lf)", star.x, star.y, i + 1, star_list[i].x, star_list[i].y);
						star.close_to_other = true;
						star_list[i].close_to_other = true;
						break;
					}
				}
			}
			
			if (res == INDIGO_OK || radius < 3) {
				star.oversaturated = lmax == max_luminance;
				star.nc_distance = sqrt((star.x - width2) * (star.x - width2) + (star.y - height2) * (star.y - height2));
				star.nc_distance /= divider;
				star.luminance = (luminance > 0) ? log(fabs(luminance)) : 0;
				star_list[found++] = star;
			}
		}
		if (found >= stars_max) {
			break;
		}
	}
	free(buf);
	
	qsort(star_list, found, sizeof(indigo_star_detection), luminance_comparator);
	
	INDIGO_DEBUG(
							 for (size_t i = 0;i < found; i++) {
								 indigo_debug(
															"%s: star #%u = (%lf, %lf), ncdist = %lf, lum = %lf, close_to_other = %d, oversaturated = %d",
															__FUNCTION__,
															i+1,
															star_list[i].x,
															star_list[i].y,
															star_list[i].nc_distance,
															star_list[i].luminance,
															star_list[i].close_to_other,
															star_list[i].oversaturated
															);
							 }
							 )
	
	*stars_found = found;
	return INDIGO_OK;
}

// ---- end of reference ----

static int checks = 0, failures = 0;

static bool same_double(double a, double b) {
	return memcmp(&a, &b, sizeof(double)) == 0;
}

static bool same_stars(const char *label, int width, int height, int radius, int stars_max, indigo_result r1, int n1, const indigo_star_detection *s1, indigo_result r2, int n2, const indigo_star_detection *s2) {
	checks++;
	bool ok = r1 == r2 && n1 == n2;
	for (int i = 0; ok && i < n1; i++) {
		ok = same_double(s1[i].x, s2[i].x) && same_double(s1[i].y, s2[i].y) && same_double(s1[i].nc_distance, s2[i].nc_distance) && same_double(s1[i].luminance, s2[i].luminance) && s1[i].oversaturated == s2[i].oversaturated && s1[i].close_to_other == s2[i].close_to_other;
		if (!ok) {
			printf("  star #%d differs: (%.17g, %.17g) lum %.17g sat %d close %d / (%.17g, %.17g) lum %.17g sat %d close %d\n", i, s1[i].x, s1[i].y, s1[i].luminance, s1[i].oversaturated, s1[i].close_to_other, s2[i].x, s2[i].y, s2[i].luminance, s2[i].oversaturated, s2[i].close_to_other);
		}
	}
	if (!ok) {
		failures++;
		printf("FAIL %s %dx%d radius %d stars_max %d: result %d/%d, stars %d/%d\n", label, width, height, radius, stars_max, r1, r2, n1, n2);
	}
	return ok;
}

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void compare(const char *label, indigo_raw_type type, void *data, int width, int height, int radius, int stars_max, double *reference_ms, double *library_ms) {
	indigo_star_detection *expected = calloc(stars_max, sizeof(indigo_star_detection));
	indigo_star_detection *actual = calloc(stars_max, sizeof(indigo_star_detection));
	int n1 = 0, n2 = 0;
	double t0 = now_ms();
	indigo_result r1 = reference_find_stars_precise(type, data, radius, width, height, stars_max, expected, &n1);
	double t1 = now_ms();
	indigo_result r2 = indigo_find_stars_precise(type, data, radius, width, height, stars_max, actual, &n2);
	double t2 = now_ms();
	if (reference_ms) *reference_ms = t1 - t0;
	if (library_ms) *library_ms = t2 - t1;
	same_stars(label, width, height, radius, stars_max, r1, n1, expected, r2, n2, actual);
	free(expected);
	free(actual);
}

enum { ASTRO_EMPTY, ASTRO_FEW, ASTRO, DENSE, CONSTANT, SATURATED, UNIFORM, BRIGHT_PIXELS, LINES, KINDS };
static const char *kind_names[KINDS] = { "empty", "few", "astro", "dense", "constant", "saturated", "uniform", "bright pixels", "lines" };

static void make(int kind, int width, int height, uint64_t seed, uint16_t **data16, uint8_t **data8) {
	size_t size = (size_t)width * height;
	int stars = kind == ASTRO_EMPTY ? 0 : kind == ASTRO_FEW ? 5 : kind == DENSE ? (int)(size / 1500) : (int)(size / 20000) + 3;
	*data16 = test_make_bayer16(width, height, seed, stars);
	*data8 = test_make_bayer8(width, height, seed, stars);
	test_rng rng = { seed * 7919 + kind };
	for (size_t i = 0; i < size; i++) {
		switch (kind) {
			case CONSTANT: (*data16)[i] = 1234; (*data8)[i] = 77; break;
			case SATURATED: (*data16)[i] = 65535; (*data8)[i] = 255; break;
			case UNIFORM: (*data16)[i] = test_rng_u32(&rng) & 0xFFFF; (*data8)[i] = test_rng_u32(&rng) & 0xFF; break;
			case BRIGHT_PIXELS:
				if (test_rng_u32(&rng) % 10 == 0) { (*data16)[i] = 30000 + test_rng_u32(&rng) % 35536; (*data8)[i] = 128 + test_rng_u32(&rng) % 128; }
				else { (*data16)[i] = test_rng_u32(&rng) % 500; (*data8)[i] = test_rng_u32(&rng) % 4; }
				break;
			default: break;
		}
	}
	if (kind == LINES) { // hot rows / columns must not be reported as stars, real stars next to them must
		for (int x = 0; x < width; x++) { (*data16)[(size_t)(height / 3) * width + x] = 50000; (*data8)[(size_t)(height / 3) * width + x] = 230; }
		for (int y = 0; y < height; y++) { (*data16)[(size_t)y * width + width / 2] = 50000; (*data8)[(size_t)y * width + width / 2] = 230; }
	}
}

int main() {
	const int sizes[][2] = { { 96, 80 }, { 320, 240 }, { 640, 480 }, { 1024, 768 }, { 1001, 803 } };
	const int radii[] = { 0, 3, 4, 8 };
	const int maxima[] = { 1, 7, 100, 1000 };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		int width = sizes[s][0], height = sizes[s][1];
		for (int kind = 0; kind < KINDS; kind++) {
			uint16_t *data16;
			uint8_t *data8;
			make(kind, width, height, 10 + s, &data16, &data8);
			for (unsigned r = 0; r < sizeof(radii) / sizeof(radii[0]); r++) {
				for (unsigned m = 0; m < sizeof(maxima) / sizeof(maxima[0]); m++) {
					// the bright pixel / uniform frames have hundreds of thousands of peaks, keep the reference affordable
					if ((kind == BRIGHT_PIXELS || kind == UNIFORM) && (maxima[m] > 7 || width * height > 400000)) continue;
					char label[64];
					snprintf(label, sizeof(label), "%s 16 bit", kind_names[kind]);
					compare(label, INDIGO_RAW_MONO16, data16, width, height, radii[r], maxima[m], NULL, NULL);
					snprintf(label, sizeof(label), "%s 8 bit", kind_names[kind]);
					compare(label, INDIGO_RAW_MONO8, data8, width, height, radii[r], maxima[m], NULL, NULL);
				}
			}
			free(data16);
			free(data8);
		}
	}
	// timing on a full frame with 600 stars, radius 4, 100 stars requested
	{
		const int width = 4144, height = 2822;
		uint16_t *frame = test_make_bayer16(width, height, 1, 600);
		double reference[3], library[3];
		for (int run = 0; run < 3; run++) {
			compare("full frame", INDIGO_RAW_MONO16, frame, width, height, 4, 100, &reference[run], &library[run]);
		}
		printf("timing 4144x2822 16 bit, 100 stars, radius 4, best of 3: original %.0f ms, new %.0f ms\n", fmin(fmin(reference[0], reference[1]), reference[2]), fmin(fmin(library[0], library[1]), library[2]));
		free(frame);
	}
	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
