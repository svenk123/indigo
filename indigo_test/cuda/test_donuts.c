// Regression test for indigo_donuts_frame_digest[_clipped](): the MONO8/MONO16 paths were changed to cache the
// (deterministic, side-effect-free) clear_hot_pixel_8/16() result from the first pass instead of recomputing it
// identically in the second pass. Verified once, manually, against the pre-optimization implementation (git
// history) on the frames below - the checksums here are those confirmed-matching values, pinned so a future
// change that alters the result gets caught. RGB* paths are unaffected by that change (no clear_hot_pixel call)
// and are covered here for basic sanity only.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>

#include "test_image.h"

static int checks = 0, failures = 0;

static uint64_t fnv(const void *data, size_t n) {
	const uint8_t *p = data;
	uint64_t h = 1469598103934665603ULL;
	for (size_t i = 0; i < n; i++) {
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	return h;
}

static void check_golden(const char *label, indigo_raw_type type, const void *data, int width, int height, int edge_clipping, int expected_digest_width, int expected_digest_height, double expected_snr, uint64_t expected_fft_x_hash, uint64_t expected_fft_y_hash) {
	indigo_frame_digest digest;
	memset(&digest, 0, sizeof(digest));
	indigo_result r = indigo_donuts_frame_digest(type, data, width, height, edge_clipping, &digest);
	checks++;
	bool ok = r == INDIGO_OK && digest.width == expected_digest_width && digest.height == expected_digest_height && digest.snr == expected_snr;
	uint64_t fft_x_hash = 0, fft_y_hash = 0;
	int digest_width = digest.width, digest_height = digest.height;
	double snr = digest.snr;
	if (r == INDIGO_OK) {
		fft_x_hash = fnv(digest.fft_x, 2 * digest.width * sizeof(double));
		fft_y_hash = fnv(digest.fft_y, 2 * digest.height * sizeof(double));
		ok = ok && fft_x_hash == expected_fft_x_hash && fft_y_hash == expected_fft_y_hash;
		indigo_delete_frame_digest(&digest);
	}
	if (!ok) {
		failures++;
		printf("FAIL %-20s result=%d (expected %d) digest=%dx%d (expected %dx%d) snr=%.17g (expected %.17g) fft_x=%016lx (expected %016lx) fft_y=%016lx (expected %016lx)\n", label, r, INDIGO_OK, digest_width, digest_height, expected_digest_width, expected_digest_height, snr, expected_snr, (unsigned long)fft_x_hash, (unsigned long)expected_fft_x_hash, (unsigned long)fft_y_hash, (unsigned long)expected_fft_y_hash);
	} else {
		printf("ok   %-20s digest=%dx%d snr=%.6g\n", label, digest_width, digest_height, snr);
	}
}

static void check_no_crash(const char *label, indigo_raw_type type, const void *data, int width, int height) {
	indigo_frame_digest digest;
	memset(&digest, 0, sizeof(digest));
	indigo_result r = indigo_donuts_frame_digest(type, data, width, height, 8, &digest);
	checks++;
	bool ok = r == INDIGO_OK && digest.width > 0 && digest.height > 0 && isfinite(digest.snr);
	printf("%s %-20s result=%d digest=%dx%d snr=%.6g\n", ok ? "ok  " : "FAIL", label, r, digest.width, digest.height, digest.snr);
	if (!ok) failures++;
	if (r == INDIGO_OK) indigo_delete_frame_digest(&digest);
}

int main() {
	// golden values: confirmed bit-identical against the pre-optimization implementation for these exact frames
	{
		uint16_t *img = test_make_bayer16(4144, 2822, 3, 4144 * 2822 / 20000 + 5);
		check_golden("12 MP mono16", INDIGO_RAW_MONO16, img, 4144, 2822, 8, 8192, 4096, 73.116189127424946, 0xb009e92009f4bef9ULL, 0x8a0136b99ecbfa6aULL);
		uint8_t *img8 = malloc((size_t)4144 * 2822);
		for (size_t i = 0; i < (size_t)4144 * 2822; i++) img8[i] = img[i] >> 8;
		check_golden("12 MP mono8", INDIGO_RAW_MONO8, img8, 4144, 2822, 8, 8192, 4096, 65.003504681499265, 0xc4ea179344830524ULL, 0x93be8ed123367a02ULL);
		free(img);
		free(img8);
	}
	{
		uint16_t *img = test_make_bayer16(500, 500, 3, 500 * 500 / 20000 + 5);
		check_golden("500x500 mono16", INDIGO_RAW_MONO16, img, 500, 500, 8, 512, 512, 89.763269570991781, 0x3049196ac86a6627ULL, 0x05dad90bc93b2ef3ULL);
		free(img);
	}
	{
		uint16_t *img16 = test_make_bayer16(300, 300, 3, 300 * 300 / 20000 + 5);
		uint8_t *img = malloc(300 * 300);
		for (size_t i = 0; i < 300 * 300; i++) img[i] = img16[i] >> 8;
		check_golden("300x300 mono8", INDIGO_RAW_MONO8, img, 300, 300, 8, 512, 512, 136.44083046970883, 0x31673dfc73e48da5ULL, 0x9aa77114bd60764cULL);
		free(img16);
		free(img);
	}

	// RGB* paths - unaffected by the cache change (no clear_hot_pixel call), sanity only
	{
		int w = 640, h = 480;
		uint16_t *rgb48 = test_make_rgb48(w, h, 5, w * h / 20000 + 5);
		uint8_t *rgb24 = malloc((size_t)w * h * 3);
		uint8_t *rgba32 = malloc((size_t)w * h * 4);
		uint8_t *abgr32 = malloc((size_t)w * h * 4);
		for (size_t i = 0; i < (size_t)w * h; i++) {
			unsigned r = rgb48[3 * i] >> 8, g = rgb48[3 * i + 1] >> 8, b = rgb48[3 * i + 2] >> 8;
			rgb24[3 * i] = r; rgb24[3 * i + 1] = g; rgb24[3 * i + 2] = b;
			rgba32[4 * i] = r; rgba32[4 * i + 1] = g; rgba32[4 * i + 2] = b; rgba32[4 * i + 3] = 255;
			abgr32[4 * i] = 255; abgr32[4 * i + 1] = b; abgr32[4 * i + 2] = g; abgr32[4 * i + 3] = r;
		}
		check_no_crash("rgb24", INDIGO_RAW_RGB24, rgb24, w, h);
		check_no_crash("rgba32", INDIGO_RAW_RGBA32, rgba32, w, h);
		check_no_crash("abgr32", INDIGO_RAW_ABGR32, abgr32, w, h);
		check_no_crash("rgb48", INDIGO_RAW_RGB48, rgb48, w, h);
		free(rgb48); free(rgb24); free(rgba32); free(abgr32);
	}

	// no-signal (flat) frame: must be rejected with INDIGO_GUIDE_ERROR, not crash or leak (checked separately
	// with the process exit status of the whole test binary, nothing more to assert here)
	{
		int w = 200, h = 200;
		uint16_t *flat = calloc((size_t)w * h, sizeof(uint16_t));
		for (size_t i = 0; i < (size_t)w * h; i++) flat[i] = 1000;
		indigo_frame_digest digest;
		memset(&digest, 0, sizeof(digest));
		indigo_result r = indigo_donuts_frame_digest(INDIGO_RAW_MONO16, flat, w, h, 8, &digest);
		checks++;
		bool ok = r == INDIGO_GUIDE_ERROR;
		printf("%s %-20s result=%d (expected INDIGO_GUIDE_ERROR=%d)\n", ok ? "ok  " : "FAIL", "flat (no signal)", r, INDIGO_GUIDE_ERROR);
		if (!ok) failures++;
		free(flat);
	}

	printf("%s: %d checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
	return failures ? 1 : 0;
}
