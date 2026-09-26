// Runs indigo_raw_to_jpeg() (the preview of every exposure) on a minimal fake CCD device, decodes the JPEG and
// compares it with the image produced by the stretch functions (PSNR). Covers 8/16 bit mono, all Bayer patterns,
// 24 and 48 bit RGB, odd sizes, and the optional histogram.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_ccd_driver.h>
#include <indigo/indigo_stretch.h>

#include "test_image.h"

static int checks = 0, failures = 0;

#define MIN_PSNR 36.0

static double psnr_of_jpeg(const unsigned char *jpeg, unsigned long size, const unsigned char *reference, int width, int height, int components, bool *dimensions_ok) {
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, (unsigned char *)jpeg, size);
	jpeg_read_header(&cinfo, TRUE);
	cinfo.out_color_space = components == 1 ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_start_decompress(&cinfo);
	*dimensions_ok = (int)cinfo.output_width == width && (int)cinfo.output_height == height && (int)cinfo.output_components == components;
	double sum = 0;
	if (*dimensions_ok) {
		unsigned char *row = malloc((size_t)width * components);
		while (cinfo.output_scanline < cinfo.output_height) {
			int y = cinfo.output_scanline;
			jpeg_read_scanlines(&cinfo, &row, 1);
			for (int i = 0; i < width * components; i++) {
				double d = (double)row[i] - reference[(size_t)y * width * components + i];
				sum += d * d;
			}
		}
		free(row);
		jpeg_finish_decompress(&cinfo);
	}
	jpeg_destroy_decompress(&cinfo);
	double mse = sum / ((double)width * height * components);
	return mse == 0 ? 99 : 10 * log10(255.0 * 255.0 / mse);
}

// reference: the stretch functions the preview uses
static unsigned char *reference_image(int bpp, const char *bayer, const void *data, int width, int height, double B, double C, int *components) {
	unsigned char *out = calloc((size_t)width * height * 3, 1);
	double sh[3], mi[3], hi[3];
	unsigned long *hist[3] = { NULL, NULL, NULL }, tot[3];
	const int sample_by = width < 0x1FF ? 1 : width / 0x1FF;
	*components = 3;
	if (bpp == 16 && bayer) {
		#define CASE16(name, fn) if (!strcmp(bayer, name)) { indigo_compute_stretch_params_16_##fn(data, width, height, sample_by, sh, mi, hi, hist, tot, B, C); indigo_stretch_16_##fn(data, width, height, out, sh, mi, hi, tot); }
		CASE16("RGGB", rggb) else CASE16("GBRG", gbrg) else CASE16("GRBG", grbg) else CASE16("BGGR", bggr)
	} else if (bpp == 16) {
		*components = 1;
		indigo_compute_stretch_params_16(data, width, height, sample_by, sh, mi, hi, hist, B, C);
		indigo_stretch_16(data, width, height, out, sh, mi, hi);
	} else if (bpp == 8 && bayer) {
		#define CASE8(name, fn) if (!strcmp(bayer, name)) { indigo_compute_stretch_params_8_##fn(data, width, height, sample_by, sh, mi, hi, hist, tot, B, C); indigo_stretch_8_##fn(data, width, height, out, sh, mi, hi, tot); }
		CASE8("RGGB", rggb) else CASE8("GBRG", gbrg) else CASE8("GRBG", grbg) else CASE8("BGGR", bggr)
	} else if (bpp == 8) {
		*components = 1;
		indigo_compute_stretch_params_8(data, width, height, sample_by, sh, mi, hi, hist, B, C);
		indigo_stretch_8(data, width, height, out, sh, mi, hi);
	} else if (bpp == 24) {
		indigo_compute_stretch_params_24(data, width, height, sample_by, sh, mi, hi, hist, tot, B, C);
		indigo_stretch_24(data, width, height, out, sh, mi, hi, tot);
	} else if (bpp == 48) {
		indigo_compute_stretch_params_48(data, width, height, sample_by, sh, mi, hi, hist, tot, B, C);
		indigo_stretch_48(data, width, height, out, sh, mi, hi, tot);
	}
	for (int i = 0; i < 3; i++) free(hist[i]);
	return out;
}

static void check(const char *label, int bpp, const char *bayer, const void *data, int width, int height, bool histogram) {
	static indigo_device device_storage;
	indigo_device *device = &device_storage; // the CCD_* macros expect a pointer called device
	static bool initialised = false;
	if (!initialised) { // just what indigo_raw_to_jpeg() reads: the JPEG quality
		device->device_context = calloc(1, sizeof(indigo_ccd_context));
		CCD_JPEG_SETTINGS_PROPERTY = indigo_init_number_property(NULL, "fake", CCD_JPEG_SETTINGS_PROPERTY_NAME, "", "", INDIGO_OK_STATE, INDIGO_RW_PERM, 3);
		CCD_JPEG_SETTINGS_QUALITY_ITEM->number.target = 90;
		initialised = true;
	}
	const double B = 0.25, C = -2.8;
	void *jpeg = NULL, *histogram_jpeg = NULL;
	unsigned long jpeg_size = 0, histogram_size = 0;
	indigo_raw_to_jpeg(device, (void *)data, width, height, bpp, bayer, &jpeg, &jpeg_size, histogram ? &histogram_jpeg : NULL, histogram ? &histogram_size : NULL, B, C);
	checks++;
	int components;
	unsigned char *reference = reference_image(bpp, bayer, data, width, height, B, C, &components);
	bool dimensions_ok = false;
	double psnr = jpeg && jpeg_size ? psnr_of_jpeg(jpeg, jpeg_size, reference, width, height, components, &dimensions_ok) : 0;
	bool ok = jpeg && jpeg_size > 0 && dimensions_ok && psnr >= MIN_PSNR && (!histogram || (histogram_jpeg && histogram_size > 0));
	printf("%s %-34s %5dx%-5d %8lu bytes, PSNR %5.2f dB\n", ok ? "ok  " : "FAIL", label, width, height, jpeg_size, psnr);
	if (!ok) failures++;
	free(reference);
	free(jpeg);
	free(histogram_jpeg);
}

// crop of a larger RGGB frame: offset (0,0) is RGGB, (0,1) GBRG, (1,0) GRBG, (1,1) BGGR
static void *crop(const void *source, int source_width, int bytes, int dx, int dy, int width, int height) {
	unsigned char *out = malloc((size_t)width * height * bytes);
	for (int y = 0; y < height; y++) {
		memcpy(out + (size_t)y * width * bytes, (const unsigned char *)source + ((size_t)(y + dy) * source_width + dx) * bytes, (size_t)width * bytes);
	}
	return out;
}

int main() {
	const int sizes[][2] = { { 1024, 768 }, { 1001, 803 }, { 640, 481 }, { 4144, 2822 } };
	static const char *patterns[] = { "RGGB", "GBRG", "GRBG", "BGGR" };
	static const int offsets[][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		int width = sizes[s][0], height = sizes[s][1];
		bool large = width > 2000;
		uint16_t *big16 = test_make_bayer16(width + 2, height + 2, 5 + s, 200);
		uint8_t *big8 = test_make_bayer8(width + 2, height + 2, 5 + s, 200);
		char label[64];
		for (int p = 0; p < (large ? 1 : 4); p++) {
			uint16_t *data16 = crop(big16, width + 2, 2, offsets[p][0], offsets[p][1], width, height);
			uint8_t *data8 = crop(big8, width + 2, 1, offsets[p][0], offsets[p][1], width, height);
			snprintf(label, sizeof(label), "16 bit %s", patterns[p]);
			check(label, 16, patterns[p], data16, width, height, p == 0);
			snprintf(label, sizeof(label), "8 bit %s", patterns[p]);
			check(label, 8, patterns[p], data8, width, height, false);
			if (p == 0) {
				check("16 bit mono", 16, NULL, data16, width, height, true);
				check("8 bit mono", 8, NULL, data8, width, height, false);
			}
			free(data16);
			free(data8);
		}
		if (!large) {
			uint16_t *rgb48 = test_make_rgb48(width, height, 5 + s, 200);
			uint8_t *rgb24 = malloc((size_t)width * height * 3);
			for (size_t i = 0; i < (size_t)width * height * 3; i++) { unsigned v = rgb48[i] >> 4; rgb24[i] = v > 255 ? 255 : v; }
			check("24 bit RGB", 24, NULL, rgb24, width, height, false);
			check("48 bit RGB", 48, NULL, rgb48, width, height, false);
			free(rgb48);
			free(rgb24);
		}
		free(big16);
		free(big8);
	}
	printf("%s: %d checks, %d failures (PSNR >= %.0f dB against the stretched image)\n", failures ? "FAILED" : "PASSED", checks, failures, MIN_PSNR);
	return failures ? 1 : 0;
}
