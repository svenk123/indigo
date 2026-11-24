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
	if (width == 1 || height == 1) {
		float sums[3] = { 0, 0, 0 };
		int counts[3] = { 0, 0, 0 };
		const int first_row = row > 0 ? row - 1 : 0, last_row = row < height - 1 ? row + 1 : height - 1;
		const int first_column = column > 0 ? column - 1 : 0, last_column = column < width - 1 ? column + 1 : width - 1;
		for (int y = first_row; y <= last_row; y++) {
			for (int x = first_column; x <= last_column; x++) {
				const int color = offsets ^ ((x & 1) << 4 | (y & 1));
				const int channel = color == 0 ? 0 : color == 0x11 ? 2 : 1;
				sums[channel] += raw[y * width + x];
				counts[channel]++;
			}
		}
		red = counts[0] ? sums[0] / counts[0] : raw[index];
		green = counts[1] ? sums[1] / counts[1] : raw[index];
		blue = counts[2] ? sums[2] / counts[2] : raw[index];
		return;
	}
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
					blue = raw[index + width - 1];
				} else if (row == height - 1) {
					green = (raw[index - 1] + raw[index - width]) / 2.0f;
					blue = raw[index - width - 1];
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
					red = (raw[index - width + 1] + raw[index + width + 1]) / 2.0f;
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
// k1, k2 - rational transfer coefficients without division by the midpoint term

static INDIGO_HD inline uint8_t indigo_stretch_value(float value, int native_shadows, int native_highlights, float k1, float k2, float midtones) {
	if (value <= native_shadows) {
		return 0;
	} else if (value >= native_highlights) {
		return 0xFF;
	} else if (midtones == 0) {
		return 0xFF;
	} else if (midtones == 1) {
		return 0;
	} else {
		const float input_floored = (value - native_shadows);
		return (uint8_t)(k1 * input_floored / (k2 * input_floored - midtones));
	}
}

// Constants of the stretch of one channel, computed exactly like the CPU code does it

typedef struct {
	int native_shadows, native_highlights;
	float k1, k2, midtones;
} indigo_stretch_channel;

static inline indigo_stretch_channel indigo_make_stretch_channel(double shadows, double midtones, double highlights, double max_input) {
	indigo_stretch_channel channel;
	const float hs_range_factor = highlights == shadows ? 1.0f : 1.0f / (highlights - shadows);
	channel.native_shadows = (int)(shadows * max_input);
	channel.native_highlights = (int)(highlights * max_input);
	channel.k1 = (midtones - 1) * hs_range_factor * 0xFF / max_input;
	channel.k2 = ((2 * midtones) - 1) * hs_range_factor / max_input;
	channel.midtones = midtones;
	return channel;
}

#endif
