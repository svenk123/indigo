// Copyright (c) 2023 Rumen G. Bogdanovski
// All rights reserved.
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

// Pixel functions of the preview stretch, shared by the CPU code (indigo_stretch.cpp) and the CUDA kernels
// (indigo_stretch_cuda.cu) so that both produce exactly the same pixels. Internal header, not installed.

#ifndef indigo_stretch_kernels_h
#define indigo_stretch_kernels_h

#include <stdint.h>

#ifdef __CUDACC__
#define INDIGO_HD __host__ __device__
#else
#define INDIGO_HD
#endif

// raw - raw pixels, any unsigned int
// index - offset of the current pixel in raw
// row, column - row, column of the current pixel (to skip line + to detect edges of the frame)
// width, height - width, height of the frame
// offsets - 0x00 = RGGB, 0x01 = GBRG, 0x10 = GRBG, 0x11 = BGGR
// red, gree, blue - debayered pixel as an mean value of 1 to 4 pixels

template <typename T> static INDIGO_HD inline void indigo_debayer_pixel(const T *raw, int index, int row, int column, int width, int height, int offsets, float &red, float &green, float &blue) {
	switch (offsets ^ ((column & 1) << 4 | (row & 1))) {
		case 0x00:
			red = raw[index];
			if (column == 0) {
				if (row == 0) {
					green = (raw[index + 1] + raw[index + width]) / 2.0f;
					blue = raw[index + width + 1];
				} else if (row == height - 1) {
					green = (raw[index + 1] + raw[index - width]) / 2.0f;
					blue = raw[index - width + 1];
				} else {
					green = (raw[index + 1] + raw[index + width] + raw[index - width]) / 3.0f;
					blue = (raw[index - width + 1] + raw[index + width + 1]) / 2.0f;
				}
			} else if (column == width - 1) {
				if (row == 0) {
					green = (raw[index - 1] + raw[index + width]) / 2.0f;
					blue = (raw[index + width - 1] + raw[index + width + 1]) / 2.0f;
				} else if (row == height - 1) {
					green = (raw[index - 1] + raw[index - width]) / 2.0f;
					blue = (raw[index - width - 1] + raw[index - width + 1]) / 2.0f;
				} else {
					green = (raw[index - 1] + raw[index + width] + raw[index - width]) / 3.0f;
					blue = (raw[index - width - 1] + raw[index + width - 1]) / 2.0f;
				}
			} else {
				if (row == 0) {
					green = (raw[index + 1] + raw[index - 1] + raw[index + width]) / 3.0f;
					blue = (raw[index + width - 1] + raw[index + width + 1]) / 2.0f;
				} else if (row == height - 1) {
					green = (raw[index + 1] + raw[index - 1] + raw[index - width]) / 3.0f;
					blue = (raw[index - width - 1] + raw[index - width + 1]) / 2.0f;
				} else {
					green = (raw[index + 1] + raw[index - 1] + raw[index + width] + raw[index - width]) / 4.0f;
					blue = (raw[index - width - 1] + raw[index - width + 1] + raw[index + width - 1] + raw[index + width + 1]) / 4.0f;
				}
			}
			break;
		case 0x10:
			if (column == 0) {
				red = raw[index + 1];
			} else if (column == width - 1) {
				red = raw[index - 1];
			} else {
				red = (raw[index - 1] + raw[index + 1]) / 2.0f;
			}
			green = raw[index];
			if (row == 0) {
				blue = raw[index + width];
			} else if (row == height - 1) {
				blue = raw[index - width];
			} else {
				blue = (raw[index - width] + raw[index + width]) / 2.0f;
			}
			break;
		case 0x01:
			if (row == 0) {
				red = raw[index + width];
			} else if (row == height - 1) {
				red = raw[index - width];
			} else {
				red = (raw[index - width] + raw[index + width]) / 2.0f;
			}
			green = raw[index];
			if (column == 0) {
				blue = raw[index + 1];
			} else if (column == width - 1) {
				blue = raw[index - 1];
			} else {
				blue = (raw[index - 1] + raw[index + 1]) / 2.0f;
			}
			break;
		case 0x11:
			if (column == 0) {
				if (row == 0) {
					red = raw[index + width + 1];
					green = (raw[index + 1] + raw[index + width]) / 2.0f;
				} else if (row == height - 1) {
					red = raw[index - width + 1];
					green = (raw[index + 1] + raw[index - width]) / 2.0f;
				} else {
					red = raw[index - width + 1];
					green = (raw[index + 1] + raw[index + width] + raw[index - width]) / 3.0f;
				}
			} else if (column == width - 1) {
				if (row == 0) {
					red = raw[index + width - 1];
					green = (raw[index - 1] + raw[index + width]) / 2.0f;
				} else if (row == height - 1) {
					red = raw[index - width - 1];
					green = (raw[index - 1] + raw[index - width]) / 2.0f;
				} else {
					red = (raw[index - width - 1] + raw[index + width - 1]) / 2.0f;
					green = (raw[index - 1] + raw[index + width] + raw[index - width]) / 3.0f;
				}
			} else {
				if (row == 0) {
					red = (raw[index + width - 1] + raw[index + width + 1]) / 2.0f;
					green = (raw[index + 1] + raw[index - 1] + raw[index + width]) / 3.0f;
				} else if (row == height - 1) {
					red = (raw[index - width - 1] + raw[index - width + 1]) / 2.0f;
					green = (raw[index + 1] + raw[index - 1] + raw[index - width]) / 3.0f;
				} else {
					red = (raw[index - width - 1] + raw[index - width + 1] + raw[index + width - 1] + raw[index + width + 1]) / 4.0f;
					green = (raw[index + 1] + raw[index - 1] + raw[index + width] + raw[index - width]) / 4.0f;
				}
			}
			blue = raw[index];
			break;
	}
}

// value - pixel value
// native_shadows, native_highlights - shadows and highlights thresholds in input range
// k1_k2 = k1 * k2
// midtones_k2 = midtones / k2

static INDIGO_HD inline uint8_t indigo_stretch_value(float value, int native_shadows, int native_highlights, float k1_k2, float midtones_k2) {
	if (value < native_shadows) {
		return 0;
	} else if (value > native_highlights) {
		return 0xFF;
	} else {
		const float input_floored = (value - native_shadows);
		return k1_k2 * input_floored / (input_floored - midtones_k2);
	}
}

// Constants of the stretch of one channel, computed exactly like the CPU code does it

typedef struct {
	int native_shadows, native_highlights;
	float k1_k2, midtones_k2;
} indigo_stretch_channel;

static inline indigo_stretch_channel indigo_make_stretch_channel(double shadows, double midtones, double highlights, double max_input) {
	indigo_stretch_channel channel;
	const float hs_range_factor = highlights == shadows ? 1.0f : 1.0f / (highlights - shadows);
	channel.native_shadows = shadows * max_input;
	channel.native_highlights = highlights * max_input;
	const float k1 = (midtones - 1) * hs_range_factor * 0xFF / max_input;
	const float k2 = ((2 * midtones) - 1) * hs_range_factor / max_input;
	channel.k1_k2 = k1 / k2;
	channel.midtones_k2 = midtones / k2;
	return channel;
}

#endif
