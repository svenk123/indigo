# Tests and benchmarks for the (optional) CUDA acceleration and for the CPU optimisations that precede it.
#
#	make test               run the tests against build/lib/libindigo.a
#	make bench              CPU baseline / GPU comparison on realistic synthetic frames
#
# The tests follow the configuration libindigo was built with (make CUDA=yes / CUDA=no at the top level).

include ../../Makefile.inc

CXX ?= g++
INCLUDES = -I$(INDIGO_ROOT)/indigo_libs -I$(BUILD_INCLUDE)
DEFINES = -DINDIGO_LINUX $(if $(INDIGO_CUDA),-DINDIGO_CUDA)
OPT = -O2 -g -pthread
LIBS = $(BUILD_LIB)/libindigo.a $(BUILD_LIB)/libjpeg.a $(BUILD_LIB)/libtiff.a $(BUILD_LIB)/libtiffxx.a $(BUILD_LIB)/libraw.a $(BUILD_LIB)/libhidapi-libusb.a \
	$(filter -L% -lcudart,$(LDFLAGS)) -lstdc++ -lm -lrt -lusb-1.0 -lz -ldl

TESTS = test_stretch_params test_find_stars test_cuda_api
BENCHES = bench_cpu bench_jpeg

.PHONY: all test bench clean

all: $(TESTS) $(BENCHES)

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

bench: $(BENCHES)
	@for b in $(BENCHES); do echo "== $$b"; ./$$b; done

test_stretch_params: test_stretch_params.cpp test_image.h $(BUILD_LIB)/libindigo.a
	$(CXX) $(OPT) $(DEFINES) $(INCLUDES) -std=gnu++11 -o $@ $< $(LIBS)

test_find_stars: test_find_stars.c test_image.h $(BUILD_LIB)/libindigo.a
	$(CC) $(OPT) $(DEFINES) $(INCLUDES) -std=gnu11 -o $@ $< $(LIBS)

test_cuda_api: test_cuda_api.c $(BUILD_LIB)/libindigo.a
	$(CC) $(OPT) $(DEFINES) $(INCLUDES) -std=gnu11 -o $@ $< $(LIBS)

bench_jpeg: bench_jpeg.c test_image.h $(BUILD_LIB)/libindigo.a
	$(CC) $(OPT) $(DEFINES) $(INCLUDES) -std=gnu11 -o $@ $< $(LIBS)

bench_cpu: bench_cpu.c test_image.h $(BUILD_LIB)/libindigo.a
	$(CC) $(OPT) $(DEFINES) $(INCLUDES) -std=gnu11 -o $@ $< $(LIBS)

clean:
	rm -f $(TESTS) $(BENCHES)
