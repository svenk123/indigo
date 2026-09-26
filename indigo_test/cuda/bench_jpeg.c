// Speed / size / quality of the bundled libjpeg (IJG 9.2, no SIMD) for the preview JPEG of indigo_raw_to_jpeg().
// The frame is a realistic synthetic astro frame (see test_image.h), stretched exactly like the preview is.
// PSNR is measured against the uncompressed 8 bit image, so it shows what a setting costs in quality.
// Usage: bench_jpeg [width height]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <jpeglib.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_stretch.h>

#include "test_image.h"

#define RUNS 3

typedef struct {
	const char *label;
	int quality;
	J_DCT_METHOD dct;
	boolean optimize;
	int sampling; // 0 - library default, 444, 420
	boolean fancy; // do_fancy_downsampling: chroma is scaled down by a 16x16 DCT instead of the classic 2x2 average + 8x8 DCT
} setting;

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static void compress(const unsigned char *image, int width, int height, int components, const setting *s, unsigned char **out, unsigned long *out_size, int *h_sampling) {
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	*out = NULL;
	*out_size = 0;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_mem_dest(&cinfo, out, out_size);
	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = components;
	cinfo.in_color_space = components == 1 ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, s->quality, TRUE);
	cinfo.dct_method = s->dct;
	cinfo.optimize_coding = s->optimize;
	cinfo.do_fancy_downsampling = s->fancy;
	if (components == 3 && s->sampling == 444) {
		for (int i = 0; i < 3; i++) { cinfo.comp_info[i].h_samp_factor = 1; cinfo.comp_info[i].v_samp_factor = 1; }
	} else if (components == 3 && s->sampling == 420) {
		cinfo.comp_info[0].h_samp_factor = 2; cinfo.comp_info[0].v_samp_factor = 2;
		for (int i = 1; i < 3; i++) { cinfo.comp_info[i].h_samp_factor = 1; cinfo.comp_info[i].v_samp_factor = 1; }
	}
	*h_sampling = cinfo.comp_info[0].h_samp_factor;
	jpeg_start_compress(&cinfo, TRUE);
	while (cinfo.next_scanline < cinfo.image_height) {
		JSAMPROW row = (JSAMPROW)&image[(size_t)cinfo.next_scanline * width * components];
		jpeg_write_scanlines(&cinfo, &row, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

// Planar YCbCr 4:2:0 (padded to multiples of 16 by edge replication), as a GPU kernel could produce it
typedef struct {
	unsigned char *y, *cb, *cr;
	int y_stride, c_stride, y_rows, c_rows;
} planes;

static planes make_planes(const unsigned char *rgb, int width, int height) {
	planes p;
	p.y_stride = (width + 15) & ~15;
	p.y_rows = (height + 15) & ~15;
	p.c_stride = p.y_stride / 2;
	p.c_rows = p.y_rows / 2;
	p.y = malloc((size_t)p.y_stride * p.y_rows);
	p.cb = malloc((size_t)p.c_stride * p.c_rows);
	p.cr = malloc((size_t)p.c_stride * p.c_rows);
	for (int y = 0; y < p.y_rows; y++) {
		const unsigned char *row = &rgb[(size_t)(y < height ? y : height - 1) * width * 3];
		for (int x = 0; x < p.y_stride; x++) {
			const unsigned char *px = &row[3 * (x < width ? x : width - 1)];
			p.y[(size_t)y * p.y_stride + x] = (unsigned char)(0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2] + 0.5);
		}
	}
	for (int y = 0; y < p.c_rows; y++) {
		for (int x = 0; x < p.c_stride; x++) {
			double r = 0, g = 0, b = 0;
			for (int dy = 0; dy < 2; dy++) {
				int yy = 2 * y + dy; if (yy >= height) yy = height - 1;
				for (int dx = 0; dx < 2; dx++) {
					int xx = 2 * x + dx; if (xx >= width) xx = width - 1;
					const unsigned char *px = &rgb[((size_t)yy * width + xx) * 3];
					r += px[0]; g += px[1]; b += px[2];
				}
			}
			r /= 4; g /= 4; b /= 4;
			double cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.5, cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.5;
			p.cb[(size_t)y * p.c_stride + x] = cb < 0 ? 0 : cb > 255 ? 255 : (unsigned char)cb;
			p.cr[(size_t)y * p.c_stride + x] = cr < 0 ? 0 : cr > 255 ? 255 : (unsigned char)cr;
		}
	}
	return p;
}

static void compress_planes(const planes *p, int width, int height, const setting *s, unsigned char **out, unsigned long *out_size) {
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	*out = NULL;
	*out_size = 0;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_mem_dest(&cinfo, out, out_size);
	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_YCbCr;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, s->quality, TRUE);
	cinfo.dct_method = s->dct;
	cinfo.raw_data_in = TRUE; // no colour conversion, no downsampling inside libjpeg
	cinfo.do_fancy_downsampling = FALSE; // otherwise libjpeg 9 expects chroma in full resolution (16x16 DCT does the scaling)
	jpeg_start_compress(&cinfo, TRUE);
	JSAMPROW y_rows[16], cb_rows[8], cr_rows[8];
	while (cinfo.next_scanline < cinfo.image_height) {
		int line = cinfo.next_scanline;
		for (int i = 0; i < 16; i++) y_rows[i] = &p->y[(size_t)(line + i) * p->y_stride];
		for (int i = 0; i < 8; i++) {
			cb_rows[i] = &p->cb[(size_t)(line / 2 + i) * p->c_stride];
			cr_rows[i] = &p->cr[(size_t)(line / 2 + i) * p->c_stride];
		}
		JSAMPARRAY data[3] = { y_rows, cb_rows, cr_rows };
		jpeg_write_raw_data(&cinfo, data, 16);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

static double psnr(const unsigned char *original, const unsigned char *jpeg, unsigned long jpeg_size, int width, int height, int components) {
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, (unsigned char *)jpeg, jpeg_size);
	jpeg_read_header(&cinfo, TRUE);
	cinfo.out_color_space = components == 1 ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_start_decompress(&cinfo);
	unsigned char *row = malloc((size_t)width * components);
	double sum = 0;
	while (cinfo.output_scanline < cinfo.output_height) {
		int y = cinfo.output_scanline;
		jpeg_read_scanlines(&cinfo, &row, 1);
		const unsigned char *reference = &original[(size_t)y * width * components];
		for (int i = 0; i < width * components; i++) {
			double d = (double)row[i] - reference[i];
			sum += d * d;
		}
	}
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	free(row);
	double mse = sum / ((double)width * height * components);
	return mse == 0 ? 99 : 10 * log10(255.0 * 255.0 / mse);
}

static double run(const char *kind, const unsigned char *image, int width, int height, int components, const setting *settings, int count) {
	printf("\n%s\n%-34s %10s %12s %8s %9s\n", kind, "setting", "time [ms]", "size [KB]", "PSNR", "vs first");
	double first = 0;
	for (int k = 0; k < count; k++) {
		double t[RUNS];
		unsigned char *out = NULL;
		unsigned long size = 0;
		int h_sampling = 0;
		for (int run = 0; run < RUNS; run++) {
			double start = now_ms();
			compress(image, width, height, components, &settings[k], &out, &size, &h_sampling);
			t[run] = now_ms() - start;
			if (run < RUNS - 1) { free(out); out = NULL; }
		}
		qsort(t, RUNS, sizeof(double), compare_double);
		double q = psnr(image, out, size, width, height, components);
		if (k == 0) first = t[RUNS / 2];
		printf("%-34s %10.0f %12.0f %8.2f %8.2fx\n", settings[k].label, t[RUNS / 2], size / 1024.0, q, first / t[RUNS / 2]);
		free(out);
	}
	return first;
}

int main(int argc, char **argv) {
	int width = argc > 2 ? atoi(argv[1]) : 4144, height = argc > 2 ? atoi(argv[2]) : 2822;
	size_t size = (size_t)width * height;
	printf("frame %dx%d (%.1f MP), libjpeg %d\n", width, height, size / 1e6, JPEG_LIB_VERSION);
	uint16_t *frame = test_make_bayer16(width, height, 1, 600);
	unsigned char *rgb = malloc(size * 3), *gray = malloc(size);
	double shadows[3], midtones[3], highlights[3];
	unsigned long *histogram[3] = { NULL, NULL, NULL }, totals[3];
	indigo_compute_stretch_params_16_rggb(frame, width, height, width / 0x1FF, shadows, midtones, highlights, histogram, totals, 0.25, -2.8);
	indigo_stretch_16_rggb(frame, width, height, rgb, shadows, midtones, highlights, totals);
	for (int i = 0; i < 3; i++) free(histogram[i]);
	indigo_compute_stretch_params_16(frame, width, height, width / 0x1FF, shadows, midtones, highlights, histogram, 0.25, -2.8);
	indigo_stretch_16(frame, width, height, gray, shadows, midtones, highlights);
	free(histogram[0]);
	free(frame);

	const setting rgb_settings[] = {
		{ "current: q90 ISLOW default", 90, JDCT_ISLOW, FALSE, 0, TRUE },
		{ "q75", 75, JDCT_ISLOW, FALSE, 0, TRUE },
		{ "q95", 95, JDCT_ISLOW, FALSE, 0, TRUE },
		{ "q90 DCT IFAST", 90, JDCT_IFAST, FALSE, 0, TRUE },
		{ "q90 DCT FLOAT", 90, JDCT_FLOAT, FALSE, 0, TRUE },
		{ "q90 optimized Huffman tables", 90, JDCT_ISLOW, TRUE, 0, TRUE },
		{ "q90 chroma 4:2:0 (explicit)", 90, JDCT_ISLOW, FALSE, 420, TRUE },
		{ "q90 chroma 4:4:4", 90, JDCT_ISLOW, FALSE, 444, TRUE },
		{ "q90 no fancy downsampling", 90, JDCT_ISLOW, FALSE, 0, FALSE },
		{ "q90 no fancy downsampling + FLOAT", 90, JDCT_FLOAT, FALSE, 0, FALSE },
	};
	const double baseline_ms = run("RGB (colour cameras, 3 components)", rgb, width, height, 3, rgb_settings, sizeof(rgb_settings) / sizeof(rgb_settings[0]));

	// pre-converted planes: libjpeg only does DCT + Huffman
	{
		double t0 = now_ms();
		planes p = make_planes(rgb, width, height);
		double convert_ms = now_ms() - t0;
		printf("\nPre-converted YCbCr 4:2:0 planes (jpeg_write_raw_data); conversion on the CPU took %.0f ms (single thread, not part of the times below)\n", convert_ms);
		printf("%-34s %10s %12s %8s %9s\n", "setting", "time [ms]", "size [KB]", "PSNR", "vs first");
		const setting raw_settings[] = { { "q90 ISLOW", 90, JDCT_ISLOW, FALSE, 0, TRUE }, { "q90 FLOAT", 90, JDCT_FLOAT, FALSE, 0, TRUE } };
		for (int k = 0; k < 2; k++) {
			double t[RUNS];
			unsigned char *out = NULL;
			unsigned long size = 0;
			for (int run = 0; run < RUNS; run++) {
				double start = now_ms();
				compress_planes(&p, width, height, &raw_settings[k], &out, &size);
				t[run] = now_ms() - start;
				if (run < RUNS - 1) { free(out); out = NULL; }
			}
			qsort(t, RUNS, sizeof(double), compare_double);
			printf("%-34s %10.0f %12.0f %8.2f %8.2fx\n", raw_settings[k].label, t[RUNS / 2], size / 1024.0, psnr(rgb, out, size, width, height, 3), baseline_ms / t[RUNS / 2]);
			free(out);
		}
		printf("(\"vs first\" is relative to the current setting of the same run, %.0f ms)\n", baseline_ms);
		free(p.y); free(p.cb); free(p.cr);
	}
	const setting gray_settings[] = {
		{ "current: q90 ISLOW default", 90, JDCT_ISLOW, FALSE, 0, TRUE },
		{ "q90 DCT IFAST", 90, JDCT_IFAST, FALSE, 0, TRUE },
	};
	run("Grayscale (mono cameras, 1 component)", gray, width, height, 1, gray_settings, sizeof(gray_settings) / sizeof(gray_settings[0]));
	free(rgb);
	free(gray);
	return 0;
}
