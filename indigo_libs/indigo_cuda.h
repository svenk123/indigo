//
// You can use this software under the terms of 'INDIGO Astronomy
// open-source license' (see LICENSE.md).
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS 'AS IS' AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Internal interface of the optional CUDA acceleration (Jetson / Orin).
//
// The header is not installed (it lives outside of indigo_libs/indigo on purpose) and can be
// included unconditionally: without INDIGO_CUDA (make CUDA=no) all functions are inline
// stubs that report "not available", so callers just fall back to the CPU code.
//
// Environment variable INDIGO_CUDA=0 (or off / no / false) disables the GPU at run time.

#ifndef indigo_cuda_h
#define indigo_cuda_h

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <indigo/indigo_bus.h>
#include <indigo/indigo_raw_utils.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Minimal number of pixels for which a GPU call is expected to be faster than the CPU code. */
#define INDIGO_CUDA_MIN_PIXELS 0x3FFFF

/** Per caller context: a CUDA stream plus a few reusable device buffers. Opaque. */
typedef struct indigo_cuda_context indigo_cuda_context;

/** Layout of a planar YCbCr 4:2:0 image as libjpeg wants it for jpeg_write_raw_data(): the planes are padded to a multiple of
    16 pixels by replicating the last column / row, Cb and Cr have half the width and height of Y.
    The three planes are stored in one buffer: Y (y_size bytes), Cb (c_size), Cr (c_size). */
typedef struct {
	int y_stride, y_rows, c_stride, c_rows;
	size_t y_size, c_size;
} indigo_ycc420_layout;

static inline indigo_ycc420_layout indigo_ycc420_layout_for(int width, int height) {
	indigo_ycc420_layout layout;
	layout.y_stride = (width + 15) & ~15;
	layout.y_rows = (height + 15) & ~15;
	layout.c_stride = layout.y_stride / 2;
	layout.c_rows = layout.y_rows / 2;
	layout.y_size = (size_t)layout.y_stride * layout.y_rows;
	layout.c_size = (size_t)layout.c_stride * layout.c_rows;
	return layout;
}

#ifdef INDIGO_CUDA

/** True if a usable CUDA device is present (checked once, thread safe). */
extern bool indigo_cuda_available(void);
/** Device name, "" if not available. */
extern const char *indigo_cuda_device_name(void);
/** Compute capability as major * 10 + minor, 0 if not available. */
extern int indigo_cuda_compute_capability(void);
/** Takes a context from the pool (creates it on demand), NULL if CUDA is not available or the pool is exhausted. */
extern indigo_cuda_context *indigo_cuda_acquire(void);
/** Returns the context to the pool. */
extern void indigo_cuda_release(indigo_cuda_context *context);
/** Device buffer number slot (0..INDIGO_CUDA_BUFFERS-1) of at least size bytes, NULL on failure. The buffer is reused by the next call with the same slot. */
extern void *indigo_cuda_buffer(indigo_cuda_context *context, int slot, size_t size);
/** Runs a tiny kernel and verifies its result, true if the GPU works. */
extern bool indigo_cuda_selftest(void);
/** The CUDA stream (cudaStream_t) of the context. */
extern void *indigo_cuda_stream(indigo_cuda_context *context);
/** Switches the GPU on or off at run time (default on), e.g. to compare CPU and GPU results in tests. */
extern void indigo_cuda_set_enabled(bool enabled);

/* Stretch of the preview image (indigo_stretch_cuda.cu). They return false without touching the output if the GPU
   is not used (not available, frame smaller than INDIGO_CUDA_MIN_PIXELS, CUDA error), the caller then runs the CPU code.
   The results are identical to the CPU code. */
/** Number of stretch calls completed on the GPU so far (for tests). */
extern unsigned long indigo_cuda_stretch_calls(void);
extern bool indigo_cuda_debayer_stretch_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights);
extern bool indigo_cuda_debayer_stretch_16(const uint16_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights);
/* Debayer + stretch straight to planar YCbCr 4:2:0 (indigo_ycc420_layout_for(), colour conversion and chroma averaging exactly like libjpeg
   does it without fancy downsampling), for jpeg_write_raw_data(). planes must have y_size + 2 * c_size bytes. */
/* Full-image equalisation scale pass of indigo_equalize_bayer_channels() (MONO8/MONO16 only): applies the four
   already-computed per-channel scale factors in place (channel order matches the CPU code: ch1 = (x,y), ch2 = (x,y+1),
   ch3 = (x+1,y), ch4 = (x+1,y+1)). Sampling and the scale factors themselves stay on the CPU (cheap, sparse access). */
extern bool indigo_cuda_equalize_bayer_scale(indigo_raw_type raw_type, void *data, int width, int height, double ch1_scale, double ch2_scale, double ch3_scale, double ch4_scale);

/* Full indigo_binarize(): sum/sum_sq reduction (exact integers, order independent) followed by the per-pixel
   threshold. output must have width * height bytes. Result is identical to the CPU code. */
extern bool indigo_cuda_binarize(indigo_raw_type raw_type, const void *data, int width, int height, double sigma, uint8_t *output);

extern bool indigo_cuda_debayer_stretch_ycc_8(const uint8_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights);
extern bool indigo_cuda_debayer_stretch_ycc_16(const uint16_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights);
extern bool indigo_cuda_debayer_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output);
extern bool indigo_cuda_stretch_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights);
extern bool indigo_cuda_stretch_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights);
extern bool indigo_cuda_stretch_rgb_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef);
extern bool indigo_cuda_stretch_rgb_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef);

#else

static inline bool indigo_cuda_available(void) { return false; }
static inline const char *indigo_cuda_device_name(void) { return ""; }
static inline int indigo_cuda_compute_capability(void) { return 0; }
static inline indigo_cuda_context *indigo_cuda_acquire(void) { return NULL; }
static inline void indigo_cuda_release(indigo_cuda_context *context) { (void)context; }
static inline void *indigo_cuda_buffer(indigo_cuda_context *context, int slot, size_t size) { (void)context; (void)slot; (void)size; return NULL; }
static inline bool indigo_cuda_selftest(void) { return false; }
static inline void *indigo_cuda_stream(indigo_cuda_context *context) { (void)context; return NULL; }
static inline void indigo_cuda_set_enabled(bool enabled) { (void)enabled; }
static inline unsigned long indigo_cuda_stretch_calls(void) { return 0; }
static inline bool indigo_cuda_debayer_stretch_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights) { return false; }
static inline bool indigo_cuda_debayer_stretch_16(const uint16_t *input, int width, int height, int offsets, uint8_t *output, const double *shadows, const double *midtones, const double *highlights) { return false; }
static inline bool indigo_cuda_equalize_bayer_scale(indigo_raw_type raw_type, void *data, int width, int height, double ch1_scale, double ch2_scale, double ch3_scale, double ch4_scale) { return false; }
static inline bool indigo_cuda_binarize(indigo_raw_type raw_type, const void *data, int width, int height, double sigma, uint8_t *output) { return false; }
static inline bool indigo_cuda_debayer_stretch_ycc_8(const uint8_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights) { return false; }
static inline bool indigo_cuda_debayer_stretch_ycc_16(const uint16_t *input, int width, int height, int offsets, uint8_t *planes, const double *shadows, const double *midtones, const double *highlights) { return false; }
static inline bool indigo_cuda_debayer_8(const uint8_t *input, int width, int height, int offsets, uint8_t *output) { return false; }
static inline bool indigo_cuda_stretch_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights) { return false; }
static inline bool indigo_cuda_stretch_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights) { return false; }
static inline bool indigo_cuda_stretch_rgb_8(const uint8_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef) { return false; }
static inline bool indigo_cuda_stretch_rgb_16(const uint16_t *input, int width, int height, uint8_t *output, double shadows, double midtones, double highlights, const float *coef) { return false; }

#endif

/** Number of device buffers per context. */
#define INDIGO_CUDA_BUFFERS 4

#ifdef __cplusplus
}
#endif

#endif
