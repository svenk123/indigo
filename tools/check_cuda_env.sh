#!/bin/bash
#
# Environment check for the optional INDIGO CUDA build (NVIDIA Jetson / Orin, JetPack).
#
# Usage: check_cuda_env.sh [--cuda-home DIR] [--arch "53 87"] [--std c++11] [--skip-packages] <env|check>
#
#   env    quiet mode used by the top level Makefile: prints KEY=VALUE lines on stdout
#          (CUDA_OK=1 is printed last and only if all mandatory checks passed),
#          problems are reported on stderr
#   check  verbose report including a compile / link / run test of a tiny CUDA kernel,
#          exit status 0 = OK, 1 = at least one mandatory check failed
#
# --arch overrides the SM architecture detection (space separated, e.g. "87")
# --skip-packages skips the dpkg checks (e.g. CUDA installed from a runfile)

CUDA_HOME_ARG=/usr/local/cuda
ARCH_ARG=""
STD=c++11
SKIP_PKG=0
MODE=""

while [ $# -gt 0 ]; do
	case "$1" in
		--cuda-home) CUDA_HOME_ARG="$2"; shift 2;;
		--arch) ARCH_ARG="$2"; shift 2;;
		--std) STD="$2"; shift 2;;
		--skip-packages) SKIP_PKG=1; shift;;
		env|check) MODE="$1"; shift;;
		*) echo "usage: $0 [--cuda-home DIR] [--arch \"53 87\"] [--std c++11] [--skip-packages] <env|check>" >&2; exit 2;;
	esac
done
[ -n "$MODE" ] || { echo "usage: $0 [options] <env|check>" >&2; exit 2; }

FAILS=0
WARNS=0

ok()   { [ "$MODE" = check ] && echo "  [ OK ] $*"; return 0; }
warn() { WARNS=$((WARNS + 1)); [ "$MODE" = check ] && echo "  [WARN] $*"; return 0; }
fail() {
	FAILS=$((FAILS + 1))
	if [ "$MODE" = check ]; then echo "  [FAIL] $1"; [ -n "$2" ] && echo "         -> $2"; else echo "CUDA check: $1" >&2; [ -n "$2" ] && echo "            -> $2" >&2; fi
	return 0
}
section() { [ "$MODE" = check ] && { echo; echo "$*"; }; return 0; }

dpkg_ver() {
	command -v dpkg-query >/dev/null 2>&1 || return 0
	dpkg-query -W -f='${Status} ${Version}\n' "$1" 2>/dev/null | awk '$1 == "install" && $2 == "ok" && $3 == "installed" { print $4 }'
}

version_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]; }

[ "$MODE" = check ] && echo "INDIGO CUDA environment check"

#---------------------------------------------------------------------
section "Platform"
#---------------------------------------------------------------------

MACHINE=$(uname -m)
if [ "$(uname -s)" != Linux ]; then
	fail "CUDA build is supported on Linux only ($(uname -s) detected)"
fi
case "$MACHINE" in
	aarch64|x86_64) ok "architecture: $MACHINE";;
	*) fail "unsupported architecture: $MACHINE";;
esac

COMPAT=$(tr '\0' ' ' </proc/device-tree/compatible 2>/dev/null)
MODEL=$(tr -d '\0' </proc/device-tree/model 2>/dev/null)
SOC=""
BOARD=""
DEFAULT_ARCH=""
case "$COMPAT" in
	*tegra210*) SOC=tegra210; BOARD="Jetson Nano / TX1 (Maxwell)"; DEFAULT_ARCH=53;;
	*tegra186*) SOC=tegra186; BOARD="Jetson TX2 (Pascal)"; DEFAULT_ARCH=62;;
	*tegra194*) SOC=tegra194; BOARD="Jetson Xavier (Volta)"; DEFAULT_ARCH=72;;
	*tegra234*) SOC=tegra234; BOARD="Jetson Orin (Ampere)"; DEFAULT_ARCH=87;;
	*tegra264*) SOC=tegra264; BOARD="Jetson Thor (Blackwell)"; DEFAULT_ARCH=thor;;
esac

if [ -n "$SOC" ]; then
	ok "${MODEL:-Jetson} - $BOARD"
elif [ -n "$ARCH_ARG" ]; then
	warn "no Jetson device tree found, using --arch/CUDA_ARCH=$ARCH_ARG"
else
	fail "not a Jetson / Orin system (device tree: '${COMPAT:-none}')" "pass the SM architecture explicitly, e.g. make CUDA=yes CUDA_ARCH=87"
fi

if [ -r /etc/os-release ]; then
	OS_PRETTY=$(. /etc/os-release; echo "$PRETTY_NAME")
	ok "OS: $OS_PRETTY"
fi

#---------------------------------------------------------------------
section "L4T / JetPack"
#---------------------------------------------------------------------

L4T=$(dpkg_ver nvidia-l4t-core)
if [ -z "$L4T" ] && [ -r /etc/nv_tegra_release ]; then
	L4T=$(sed -n 's/^# R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)
fi
JETPACK=$(dpkg_ver nvidia-jetpack)
if [ -n "$SOC" ]; then
	if [ -n "$L4T" ]; then
		L4T_MAJOR=${L4T%%.*}
		case "$L4T_MAJOR" in
			32) JP_GEN="JetPack 4.x";;
			35) JP_GEN="JetPack 5.x";;
			36) JP_GEN="JetPack 6.x";;
			38) JP_GEN="JetPack 7.x";;
			*) JP_GEN="unknown JetPack generation";;
		esac
		ok "L4T $L4T ($JP_GEN)"
		if [ "$L4T_MAJOR" = 38 ] && [ -n "$JETPACK" ] && ! version_ge "$JETPACK" 7.2; then
			warn "nvidia-jetpack $JETPACK installed, JetPack 7.2 or newer is the tested target"
		fi
	else
		warn "L4T release not detected (nvidia-l4t-core not installed, /etc/nv_tegra_release missing)"
	fi
	if [ -n "$JETPACK" ]; then
		ok "nvidia-jetpack $JETPACK"
	else
		warn "meta package nvidia-jetpack not installed, individual packages are checked instead"
	fi
fi

#---------------------------------------------------------------------
section "nvcc / CUDA toolkit"
#---------------------------------------------------------------------

CUDA_HOME="$CUDA_HOME_ARG"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ] && [ "$CUDA_HOME_ARG" = /usr/local/cuda ] && command -v nvcc >/dev/null 2>&1; then
	NVCC=$(readlink -f "$(command -v nvcc)")
	CUDA_HOME=$(dirname "$(dirname "$NVCC")")
	warn "nvcc not found in $CUDA_HOME_ARG, using $NVCC from PATH (set CUDA_HOME to silence this)"
fi

CUDA_MAJOR=""
CUDA_MINOR=""
if [ ! -x "$NVCC" ]; then
	fail "nvcc not found (looked in $CUDA_HOME_ARG/bin and PATH)" "sudo apt install nvidia-jetpack   (or cuda-toolkit-<major>-<minor>), then export PATH=/usr/local/cuda/bin:\$PATH"
else
	CUDA_VER=$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')
	if [ -z "$CUDA_VER" ]; then
		fail "$NVCC does not run or reports no version"
	else
		CUDA_MAJOR=${CUDA_VER%%.*}
		CUDA_MINOR=${CUDA_VER##*.}
		if version_ge "$CUDA_VER" 10.2; then
			ok "nvcc $CUDA_VER ($NVCC)"
		else
			fail "nvcc $CUDA_VER is too old (10.2 or newer required)"
		fi
	fi
fi
if command -v g++ >/dev/null 2>&1; then
	ok "host compiler: $(g++ --version | head -n1)"
else
	fail "g++ not found (needed as nvcc host compiler)" "sudo apt install build-essential"
fi

# SM architectures
if [ -n "$ARCH_ARG" ]; then
	ARCHS="$ARCH_ARG"
elif [ "$DEFAULT_ARCH" = thor ]; then
	# sm_101 in CUDA 12.9, renamed to sm_110 in CUDA 13.0
	if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 13 ]; then ARCHS=110; else ARCHS=101; fi
else
	ARCHS="$DEFAULT_ARCH"
fi
ARCHS=$(echo $ARCHS | tr ',' ' ')
if [ -n "$ARCHS" ] && [ -x "$NVCC" ]; then
	for a in $ARCHS; do
		# --dryrun only validates the architecture, nothing is compiled
		if "$NVCC" --dryrun -gencode arch=compute_$a,code=sm_$a -c -x cu /dev/null -o /dev/null >/dev/null 2>&1; then
			ok "target architecture sm_$a supported by nvcc"
		else
			fail "nvcc $CUDA_VER does not support sm_$a" "old toolkits do not know new GPUs and new ones dropped old GPUs (CUDA 13 dropped Maxwell/Pascal/Volta) - use the toolkit of the matching JetPack release or set CUDA_ARCH"
		fi
	done
fi

# headers and libraries
INCDIR=""
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/$MACHINE-linux/include" "$CUDA_HOME/targets/sbsa-linux/include"; do
	[ -f "$d/cuda_runtime.h" ] && { INCDIR="$d"; break; }
done
LIBDIR=""
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/$MACHINE-linux/lib" "$CUDA_HOME/targets/sbsa-linux/lib"; do
	ls "$d"/libcudart.so* >/dev/null 2>&1 && { LIBDIR="$d"; break; }
done
PKG_SUFFIX=""
[ -n "$CUDA_MAJOR" ] && PKG_SUFFIX="$CUDA_MAJOR-$CUDA_MINOR"
if [ -n "$INCDIR" ]; then
	ok "cuda_runtime.h in $INCDIR"
else
	fail "cuda_runtime.h not found below $CUDA_HOME" "sudo apt install cuda-cudart-dev-${PKG_SUFFIX:-<major>-<minor>}"
fi
if [ -n "$LIBDIR" ]; then
	ok "libcudart in $LIBDIR"
else
	fail "libcudart.so not found below $CUDA_HOME" "sudo apt install cuda-cudart-dev-${PKG_SUFFIX:-<major>-<minor>}"
fi
LDCONFIG=$(command -v ldconfig || echo /sbin/ldconfig)
if [ -x "$LDCONFIG" ]; then
	if "$LDCONFIG" -p 2>/dev/null | grep -q 'libcudart\.so'; then
		ok "libcudart known to the dynamic linker"
	else
		warn "libcudart not in the ldconfig cache - add ${LIBDIR:-the CUDA lib dir} to /etc/ld.so.conf.d/ and run ldconfig, or libindigo will not start"
	fi
	if "$LDCONFIG" -p 2>/dev/null | grep -q 'libcuda\.so'; then
		ok "CUDA driver library (libcuda.so) found"
	else
		fail "CUDA driver library libcuda.so not found" "sudo apt install nvidia-l4t-cuda"
	fi
fi

#---------------------------------------------------------------------
section "Ubuntu packages"
#---------------------------------------------------------------------

if [ "$SKIP_PKG" = 1 ]; then
	warn "package check skipped (--skip-packages)"
elif ! command -v dpkg-query >/dev/null 2>&1; then
	warn "dpkg-query not available, package check skipped"
elif [ -z "$PKG_SUFFIX" ]; then
	warn "CUDA version unknown, package check skipped"
else
	REQUIRED="cuda-nvcc-$PKG_SUFFIX cuda-cudart-$PKG_SUFFIX cuda-cudart-dev-$PKG_SUFFIX"
	[ -n "$SOC" ] && REQUIRED="$REQUIRED nvidia-l4t-cuda"
	MISSING=""
	for p in $REQUIRED; do
		v=$(dpkg_ver "$p")
		if [ -n "$v" ]; then ok "$p $v"; else MISSING="$MISSING $p"; fi
	done
	if [ -n "$MISSING" ]; then
		fail "missing packages:$MISSING" "sudo apt install$MISSING   (or use CUDA_SKIP_PKG_CHECK=1 for a non-dpkg installation)"
	fi
	if [ -n "$(dpkg_ver build-essential)" ]; then
		ok "build-essential $(dpkg_ver build-essential)"
	else
		warn "package build-essential not installed"
	fi
fi

if [ -n "$SOC" ]; then
	if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx video; then
		ok "user is member of group 'video' (GPU device access)"
	else
		warn "user is not in group 'video' - GPU access may fail (sudo usermod -aG video \$USER, then log in again)"
	fi
fi

#---------------------------------------------------------------------
# compile / link / run test - the same tool chain as the Makefile uses
#---------------------------------------------------------------------

if [ "$MODE" = check ] && [ "$FAILS" = 0 ] && [ -n "$ARCHS" ]; then
	section "Compile / link / run test"
	TMP=$(mktemp -d)
	trap 'rm -rf "$TMP"' EXIT
	cat >"$TMP/t.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int *p) { p[threadIdx.x] = 2 * threadIdx.x; }
int main() {
	int n = 0;
	cudaError_t e = cudaGetDeviceCount(&n);
	if (e != cudaSuccess || n < 1) { printf("no CUDA device: %s\n", cudaGetErrorString(e)); return 2; }
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);
	int *d = NULL;
	if (cudaMallocManaged(&d, 32 * sizeof(int)) != cudaSuccess) { printf("cudaMallocManaged failed\n"); return 3; }
	k<<<1, 32>>>(d);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(e)); return 4; }
	if (d[31] != 62) { printf("wrong kernel result\n"); return 5; }
	printf("%s, compute capability %d.%d, %d MB\n", p.name, p.major, p.minor, (int)(p.totalGlobalMem >> 20));
	return 0;
}
EOF
	GENCODE=""
	for a in $ARCHS; do GENCODE="$GENCODE -gencode arch=compute_$a,code=sm_$a"; done
	if ! OUT=$("$NVCC" -std=$STD $GENCODE -Xcompiler -fPIC -c "$TMP/t.cu" -o "$TMP/t.o" 2>&1); then
		fail "nvcc could not compile the test kernel" "$(echo "$OUT" | head -n 3 | tr '\n' ' ')"
	elif ! OUT=$(g++ "$TMP/t.o" -o "$TMP/t" -L"$LIBDIR" -lcudart -Wl,-rpath,"$LIBDIR" 2>&1); then
		fail "linking against libcudart failed" "$(echo "$OUT" | head -n 3 | tr '\n' ' ')"
	elif ! OUT=$("$TMP/t" 2>&1); then
		fail "test kernel failed on the GPU: $OUT" "check the sm architecture ($ARCHS), group 'video' and that the GPU driver is loaded"
	else
		ok "test kernel ran on: $OUT"
	fi
fi

#---------------------------------------------------------------------

if [ "$MODE" = check ]; then
	echo
	if [ "$FAILS" = 0 ]; then
		echo "CUDA environment OK ($WARNS warning(s))"
	else
		echo "CUDA environment NOT usable: $FAILS error(s), $WARNS warning(s)"
	fi
fi

[ "$FAILS" = 0 ] || exit 1

if [ "$MODE" = env ]; then
	echo "CUDA_HOME=$CUDA_HOME"
	echo "CUDA_VERSION=$CUDA_VER"
	echo "CUDA_INCDIR=$INCDIR"
	echo "CUDA_LIBDIR=$LIBDIR"
	echo "CUDA_ARCHS=$(echo $ARCHS | tr ' ' ',')"
	echo "CUDA_OK=1"
fi
exit 0
