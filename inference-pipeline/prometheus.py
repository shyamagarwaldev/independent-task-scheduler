import os
import numpy as np
import httpx
from typing import Optional

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-operated.monitoring.svc.cluster.local:9090")

# Exact feature order must match training column order
FEATURE_ORDER = [
    # core 0
    ("cpu", "0", "idle"),    ("cpu", "0", "iowait"),  ("cpu", "0", "irq"),
    ("cpu", "0", "nice"),    ("cpu", "0", "softirq"), ("cpu", "0", "steal"),
    ("cpu", "0", "system"),  ("cpu", "0", "user"),
    # core 1
    ("cpu", "1", "idle"),    ("cpu", "1", "iowait"),  ("cpu", "1", "irq"),
    ("cpu", "1", "nice"),    ("cpu", "1", "softirq"), ("cpu", "1", "steal"),
    ("cpu", "1", "system"),  ("cpu", "1", "user"),
    # core 2
    ("cpu", "2", "idle"),    ("cpu", "2", "iowait"),  ("cpu", "2", "irq"),
    ("cpu", "2", "nice"),    ("cpu", "2", "softirq"), ("cpu", "2", "steal"),
    ("cpu", "2", "system"),  ("cpu", "2", "user"),
    # core 3
    ("cpu", "3", "idle"),    ("cpu", "3", "iowait"),  ("cpu", "3", "irq"),
    ("cpu", "3", "nice"),    ("cpu", "3", "softirq"), ("cpu", "3", "steal"),
    ("cpu", "3", "system"),  ("cpu", "3", "user"),
    # memory (6 features)
    ("mem", "used_bytes",            None),
    ("mem", "Buffers_bytes",         None),
    ("mem", "Cached_bytes",          None),
    ("mem", "MemAvailable_bytes",    None),
    ("mem", "MemFree_bytes",         None),
    ("mem", "MemTotal_bytes",        None),
]

N_FEATURES = len(FEATURE_ORDER)  # 38


def _instant_query(client: httpx.Client, promql: str) -> list[dict]:
    resp = client.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": promql},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")
    return data["data"]["result"]


def fetch_model_features() -> Optional[tuple[list[str], np.ndarray]]:
    """
    Returns (node_names, feature_matrix) where feature_matrix is (n_nodes, 38).
    Returns None if Prometheus is unreachable or returns no data.
    """
    try:
        with httpx.Client() as client:
            # --- CPU features ---
            # rate() over 1m window gives per-second average, multiply by 100 for %
            cpu_results = _instant_query(
                client,
                'rate(node_cpu_seconds_total[1m]) * 100'
            )

            # --- Memory features ---
            mem_total   = _instant_query(client, 'node_memory_MemTotal_bytes')
            mem_free    = _instant_query(client, 'node_memory_MemFree_bytes')
            mem_avail   = _instant_query(client, 'node_memory_MemAvailable_bytes')
            mem_buffers = _instant_query(client, 'node_memory_Buffers_bytes')
            mem_cached  = _instant_query(client, 'node_memory_Cached_bytes')

    except Exception as e:
        print(f"[prometheus] fetch failed: {e}")
        return None

    # --- Discover nodes from memory total (one result per node) ---
    nodes = {}  # instance_label -> node_name
    for r in mem_total:
        instance = r["metric"].get("instance", "")
        node = r["metric"].get("kubernetes_node", instance)  # prefer 'node' label if present
        nodes[instance] = node

    if not nodes:
        print("[prometheus] no nodes found in Prometheus results")
        return None

    node_names = sorted(nodes.values())
    instance_by_node = {v: k for k, v in nodes.items()}

    # --- Build CPU lookup: (instance, cpu_core, mode) -> value ---
    cpu_lookup: dict[tuple[str, str, str], float] = {}
    for r in cpu_results:
        instance = r["metric"].get("instance", "")
        cpu_core = r["metric"].get("cpu", "")
        mode = r["metric"].get("mode", "")
        cpu_lookup[(instance, cpu_core, mode)] = float(r["value"][1])

    # --- Build memory lookups: instance -> value ---
    def mem_lookup(results: list[dict]) -> dict[str, float]:
        return {
            r["metric"].get("instance", ""): float(r["value"][1])
            for r in results
        }

    mem_total_map   = mem_lookup(mem_total)
    mem_free_map    = mem_lookup(mem_free)
    mem_avail_map   = mem_lookup(mem_avail)
    mem_buffers_map = mem_lookup(mem_buffers)
    mem_cached_map  = mem_lookup(mem_cached)

    # --- Assemble feature matrix ---
    feature_matrix = np.zeros((len(node_names), N_FEATURES), dtype=np.float64)

    for row_idx, node_name in enumerate(node_names):
        instance = instance_by_node[node_name]

        for col_idx, feature in enumerate(FEATURE_ORDER):
            kind = feature[0]

            if kind == "cpu":
                _, core, mode = feature
                val = cpu_lookup.get((instance, core, mode), 0.0)

            elif kind == "mem":
                _, mem_key, _ = feature
                val = {
                    "used_bytes":         (mem_total_map.get(instance, 0.0)
                                           - mem_free_map.get(instance, 0.0)
                                           - mem_buffers_map.get(instance, 0.0)
                                           - mem_cached_map.get(instance, 0.0)),
                    "Buffers_bytes":      mem_buffers_map.get(instance, 0.0),
                    "Cached_bytes":       mem_cached_map.get(instance, 0.0),
                    "MemAvailable_bytes": mem_avail_map.get(instance, 0.0),
                    "MemFree_bytes":      mem_free_map.get(instance, 0.0),
                    "MemTotal_bytes":     mem_total_map.get(instance, 0.0),
                }[mem_key]
            else:
                val = 0.0

            feature_matrix[row_idx, col_idx] = val

    return node_names, feature_matrix


NETWORK_FEATURES = [
    'rx_drop',
    'tx_drop',
    'rx_bytes',
    'tx_bytes',
    'retSeg'
]


def fetch_network_features() -> Optional[tuple[list[str],np.ndarray]]:
    try:
        with httpx.Client() as client:
            rx_drop = _instant_query(
                client,
                'sum by(instance) (rate(node_network_receive_drop_total[1m]))'
            )
            tx_drop = _instant_query(
                client,
                'sum by(instance) (rate(node_network_transmit_drop_total[1m]))'
            )
            # rx_err = _instant_query(
            #     client,
            #     'sum by(instance) (rate(node_network_receive_errs_total[1m]))'
            # )
            # tx_err = _instant_query(
            #     client,
            #     'sum by(instance) (rate(node_network_transmit_errs_total[1m]))'
            # )
            rx_bytes = _instant_query(
                client,
                'sum by(instance) (rate(node_network_receive_bytes_total{device!="lo"}[1m]))'
            )
            tx_bytes = _instant_query(
                client,
                'sum by(instance) (rate(node_network_transmit_bytes_total{device!="lo"}[1m]))'
            )
            retSeg = _instant_query(
                client,
                'sum by(instance) (rate(node_netstat_Tcp_RetransSegs[1m]))'
            )
    except Exception as e:
        print(f"[prometheus] fetch failed for network metrics: {e}")
        return None
    
    instance_to_nodes = {}
    for r in rx_bytes:
        instance = r["metric"].get("instance", "")
        node = r["metric"].get("node", instance)  # prefer 'node' label if present
        instance_to_nodes[instance] = node
    if not instance_to_nodes:
        return None    

    node_to_instance = { v:k for k,v in instance_to_nodes.items()}
    node_names = sorted(instance_to_nodes.values())
    def net_lookup(results: list[dict]) -> dict[str, float]:
        return {
            r["metric"].get("instance", ""): float(r["value"][1])
            for r in results
        }
    rx_drop_lookup = net_lookup(rx_drop)
    tx_drop_lookup = net_lookup(tx_drop)
    rx_bytes_lookup = net_lookup(rx_bytes)
    tx_bytes_lookup = net_lookup(tx_bytes)
    retSeg_lookup = net_lookup(retSeg)
    feat_map = {
        "rx_drop" : rx_drop_lookup,
        "tx_drop" : tx_drop_lookup,
        "rx_bytes": rx_bytes_lookup,
        "tx_bytes": tx_bytes_lookup,
        "retSeg": retSeg_lookup 
    }
    network_feature = np.zeros((len(node_names),5),dtype=np.float64)
    for row_idx, node_name in enumerate(node_names):
        for col_idx, feat in enumerate(NETWORK_FEATURES):
            network_feature[row_idx,col_idx] = feat_map[feat].get(node_to_instance[node_name],0.0)
    return node_names , network_feature        



        
