// Baseline of the functions that are candidates for CUDA acceleration, measured on a realistic synthetic frame
// (see test_image.h). Reports the median of several runs. Usage: bench_cpu [width height]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jpeglib.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>
#include <indigo/indigo_stretch.h>
#include <indigo/indigo_base64.h>

#include "test_image.h"
#include "indigo_cuda.h"

#define RUNS 5

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double median(double *values, int count) {
	qsort(values, count, sizeof(double), compare_double);
	return values[count / 2];
}

#define BENCH(label, code) do { \
	double t[RUNS]; \
	for (int run = 0; run < RUNS; run++) { double start = now_ms(); code; t[run] = now_ms() - start; } \
	printf("%-44s %9.1f ms\n", label, median(t, RUNS)); \
} while (0)

static void free_histograms(unsigned long **histogram) {
	for (int i = 0; i < 3; i++) { free(histogram[i]); histogram[i] = NULL; }
}

static void jpeg_compress(const unsigned char *rgb, int width, int height, int quality) {
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	unsigned char *mem = NULL;
	unsigned long mem_size = 0;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_mem_dest(&cinfo, &mem, &mem_size);
	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	while (cinfo.next_scanline < cinfo.image_height) {
		JSAMPROW row = (JSAMPROW)&rgb[(size_t)cinfo.next_scanline * width * 3];
		jpeg_write_scanlines(&cinfo, &row, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	free(mem);
}

int main(int argc, char **argv) {
	int width = argc > 2 ? atoi(argv[1]) : 4144, height = argc > 2 ? atoi(argv[2]) : 2822;
	size_t size = (size_t)width * height;
	printf("frame %dx%d (%.1f MP), CUDA: %s\n\n", width, height, size / 1e6, indigo_cuda_available() ? indigo_cuda_device_name() : "not available / not built");
	uint16_t *img = test_make_bayer16(width, height, 1, 600);
	uint8_t *img8 = test_make_bayer8(width, height, 1, 600);
	uint8_t *out = malloc(size * 3);
	double shadows[3], midtones[3], highlights[3];
	unsigned long *histogram[3] = { NULL, NULL, NULL }, totals[3];
	int sample_by = width / 0x1FF;

	printf("-- preview pipeline (indigo_raw_to_jpeg)\n");
	BENCH("stretch params 16 bit rggb", { indigo_compute_stretch_params_16_rggb(img, width, height, sample_by, shadows, midtones, highlights, histogram, totals, 0.25, -2.8); free_histograms(histogram); });
	BENCH("stretch params 8 bit rggb", { indigo_compute_stretch_params_8_rggb(img8, width, height, sample_by, shadows, midtones, highlights, histogram, totals, 0.25, -2.8); free_histograms(histogram); });
	indigo_compute_stretch_params_16_rggb(img, width, height, sample_by, shadows, midtones, highlights, histogram, totals, 0.25, -2.8);
	free_histograms(histogram);
	BENCH("stretch + debayer 16 bit rggb -> RGB8", indigo_stretch_16_rggb(img, width, height, out, shadows, midtones, highlights, totals));
	BENCH("stretch mono 16 bit -> 8 bit", indigo_stretch_16(img, width, height, out, shadows, midtones, highlights));
	indigo_stretch_16_rggb(img, width, height, out, shadows, midtones, highlights, totals);
	BENCH("libjpeg compress RGB8, quality 90", jpeg_compress(out, width, height, 90));

	printf("\n-- image analysis (agents)\n");
	bool saturated;
	indigo_star_detection stars[100];
	int found = 0;
	BENCH("indigo_contrast", indigo_contrast(INDIGO_RAW_MONO16, img, NULL, width, height, &saturated));
	BENCH("indigo_find_stars_precise", indigo_find_stars_precise(INDIGO_RAW_MONO16, img, 4, width, height, 100, stars, &found));
	printf("%-44s %9d\n", "  stars found", found);
	BENCH("indigo_binarize", free(indigo_binarize(INDIGO_RAW_MONO16, img, width, height, 3.0)));
	BENCH("indigo_equalize_bayer_channels", indigo_equalize_bayer_channels(INDIGO_RAW_MONO16, img, width, height));
	{
		indigo_frame_digest digest;
		int roi = 256;
		BENCH("donuts digest, 256x256 region", { memset(&digest, 0, sizeof(digest)); indigo_donuts_frame_digest_clipped(INDIGO_RAW_MONO16, img, width, height, (width - roi) / 2, (height - roi) / 2, roi, roi, &digest); indigo_delete_frame_digest(&digest); });
		BENCH("donuts digest, full frame", { memset(&digest, 0, sizeof(digest)); indigo_donuts_frame_digest(INDIGO_RAW_MONO16, img, width, height, 0, &digest); indigo_delete_frame_digest(&digest); });
	}

	printf("\n-- transport\n");
	{
		char *encoded = malloc(size * 3);
		BENCH("base64_encode of the frame", base64_encode((unsigned char *)encoded, (unsigned char *)img, size * 2));
		free(encoded);
	}
	free(img);
	free(img8);
	free(out);
	return 0;
}
