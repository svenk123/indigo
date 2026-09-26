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

#ifdef __cplusplus
extern "C" {
#endif

/** Minimal number of pixels for which a GPU call is expected to be faster than the CPU code. */
#define INDIGO_CUDA_MIN_PIXELS 0x3FFFF

/** Per caller context: a CUDA stream plus a few reusable device buffers. Opaque. */
typedef struct indigo_cuda_context indigo_cuda_context;

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

#else

static inline bool indigo_cuda_available(void) { return false; }
static inline const char *indigo_cuda_device_name(void) { return ""; }
static inline int indigo_cuda_compute_capability(void) { return 0; }
static inline indigo_cuda_context *indigo_cuda_acquire(void) { return NULL; }
static inline void indigo_cuda_release(indigo_cuda_context *context) { (void)context; }
static inline void *indigo_cuda_buffer(indigo_cuda_context *context, int slot, size_t size) { (void)context; (void)slot; (void)size; return NULL; }
static inline bool indigo_cuda_selftest(void) { return false; }

#endif

/** Number of device buffers per context. */
#define INDIGO_CUDA_BUFFERS 4

#ifdef __cplusplus
}
#endif

#endif
