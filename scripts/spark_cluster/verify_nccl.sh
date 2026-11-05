#!/usr/bin/env bash
# Build NCCL + nccl-tests (sm_121) and run a 2-node all_reduce over the QSFP link.
# Run on node 1 after setup_node.sh has run on both nodes (network + ssh).
# Builds into ~/spark-nccl on BOTH nodes (same path is needed for mpirun).
set -euo pipefail

SUBNET="${SPARK_SUBNET:-192.168.100}"
N1="${SUBNET}.11"; N2="${SUBNET}.12"
WORK="$HOME/spark-nccl"
IFACE=$(cat /etc/spark-cluster-iface 2>/dev/null || echo enp1s0f0np0)
MPI_HOME=${MPI_HOME:-/usr/lib/aarch64-linux-gnu/openmpi}

build() {
  set -euo pipefail
  WORK="$1"; MPI_HOME="$2"
  sudo apt-get install -y libopenmpi-dev openmpi-bin git build-essential >/dev/null
  mkdir -p "$WORK" && cd "$WORK"
  [[ -d nccl ]] || git clone --depth 1 https://github.com/NVIDIA/nccl.git
  [[ -f nccl/build/lib/libnccl.so ]] || make -C nccl -j"$(nproc)" src.build \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
  [[ -d nccl-tests ]] || git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
  [[ -x nccl-tests/build/all_reduce_perf ]] || make -C nccl-tests -j"$(nproc)" MPI=1 \
      MPI_HOME="$MPI_HOME" CUDA_HOME=/usr/local/cuda NCCL_HOME="$WORK/nccl/build"
}

echo "== Building on node 1"; build "$WORK" "$MPI_HOME"
echo "== Building on node 2"; ssh -t "$N2" "$(declare -f build); build '$WORK' '$MPI_HOME'"

export NCCL_SOCKET_IFNAME=$IFACE UCX_NET_DEVICES=$IFACE OMPI_MCA_btl_tcp_if_include=$IFACE
export LD_LIBRARY_PATH="$WORK/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== all_reduce across $N1 + $N2 via $IFACE"
mpirun -np 2 -H "$N1:1,$N2:1" --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" \
  -x NCCL_SOCKET_IFNAME -x UCX_NET_DEVICES -x OMPI_MCA_btl_tcp_if_include -x LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  "$WORK/nccl-tests/build/all_reduce_perf" -b 1G -e 16G -f 2
echo "Look for 'NET/IB' in the NCCL INFO lines and busbw well above 10 GB/s at 16G."
