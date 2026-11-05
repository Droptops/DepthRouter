#!/usr/bin/env bash
# Bring one DGX Spark onto the two-node ConnectX-7 (QSFP) link.
# Run on BOTH Sparks:   sudo -E ./setup_node.sh 1    (first Spark)
#                       sudo -E ./setup_node.sh 2    (second Spark)
# Then, as your normal user on each node:  ./setup_node.sh ssh <1|2>
set -euo pipefail

SUBNET="${SPARK_SUBNET:-192.168.100}"
MTU="${SPARK_MTU:-9000}"
NETPLAN_FILE=/etc/netplan/40-spark-cx7.yaml

ip_for() { echo "${SUBNET}.1$1"; }
peer_of() { [[ $1 == 1 ]] && echo 2 || echo 1; }

detect_iface() {
  # Prefer the ConnectX-7 port whose link is Up (the one the QSFP cable is in).
  if command -v ibdev2netdev >/dev/null; then
    local up
    up=$(ibdev2netdev | awk '/\(Up\)/ {print $5}' | grep -E '^enp1s0f' | head -n1 || true)
    [[ -n $up ]] && { echo "$up"; return; }
  fi
  for i in /sys/class/net/enp1s0f*; do
    [[ -e $i && $(cat "$i/carrier" 2>/dev/null || echo 0) == 1 ]] && { basename "$i"; return; }
  done
  echo "ERROR: no ConnectX-7 port with link. Is the QSFP cable seated in both Sparks?" >&2
  exit 1
}

cmd_net() {
  local n=$1
  [[ $EUID -eq 0 ]] || { echo "Network step needs root: sudo -E $0 $n" >&2; exit 1; }
  local ifc; ifc=$(detect_iface)
  echo "Using interface $ifc -> $(ip_for "$n")/24 (MTU $MTU)"
  cat > "$NETPLAN_FILE" <<YAML
network:
  version: 2
  ethernets:
    $ifc:
      dhcp4: false
      mtu: $MTU
      addresses: [$(ip_for "$n")/24]
YAML
  chmod 600 "$NETPLAN_FILE"
  netplan apply
  echo "$ifc" > /etc/spark-cluster-iface
  sleep 2
  local peer; peer=$(ip_for "$(peer_of "$n")")
  if ping -c 3 -W 2 "$peer" >/dev/null; then
    echo "OK: peer $peer reachable."
  else
    echo "Peer $peer not answering yet. Run this script on the other Spark, then: ping $peer"
  fi
}

cmd_ssh() {
  local n=$1 peer; peer=$(ip_for "$(peer_of "$n")")
  [[ -f ~/.ssh/id_ed25519 ]] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
  ssh-copy-id -o StrictHostKeyChecking=accept-new "$USER@$peer"
  ssh -o BatchMode=yes "$USER@$peer" hostname && echo "OK: passwordless SSH to $peer."
}

case "${1:-}" in
  1|2) cmd_net "$1" ;;
  ssh) [[ ${2:-} == 1 || ${2:-} == 2 ]] || { echo "usage: $0 ssh <1|2>"; exit 1; }; cmd_ssh "$2" ;;
  *) echo "usage: sudo -E $0 <1|2>   |   $0 ssh <1|2>"; exit 1 ;;
esac
