# Two-Spark cluster

Connects two DGX Sparks over the ConnectX-7 QSFP link (node 1 = 192.168.100.11, node 2 = .12).

1. Plug the QSFP cable into a ConnectX-7 port on each Spark.
2. On Spark 1: `sudo -E ./setup_node.sh 1`. On Spark 2: `sudo -E ./setup_node.sh 2`.
3. Set up SSH both ways (as your user, same username on both): `./setup_node.sh ssh 1` on Spark 1, `./setup_node.sh ssh 2` on Spark 2.
4. On Spark 1: `./verify_nccl.sh`. This builds NCCL and nccl-tests on both nodes and runs a 2-rank all_reduce.

Optional, for a vLLM/Ray deployment across both nodes:

```bash
# Spark 1
ray start --head --node-ip-address=192.168.100.11 --port=6379
# Spark 2
ray start --address=192.168.100.11:6379 --node-ip-address=192.168.100.12
ray status   # should show 2 nodes, 2 GPUs
```

Override the subnet or MTU with `SPARK_SUBNET` and `SPARK_MTU`.
