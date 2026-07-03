import threading
import time
import traceback

from cache import cache
from resource_model import model
from network_model import network_score
from prometheus import fetch_model_features, fetch_network_features

INTERVAL_SECONDS = float(15)

def run_once():
    result = fetch_model_features()
    net_met = fetch_network_features()
    if result is None:
        print("[scheduler] skipping cycle: prometheus fetch returned nothing")
        return
    if net_met is None:
        print("[scheduler] skipping cycle: prometheus network metrics fetch returned nothing")
        return

    node_names1, feature_matrix = result
    node_names2, network_matrix = net_met
    if node_names1 != node_names2:
        print(
            "[scheduler] node ordering mismatch:\n"
            f"resource={node_names1}\n"
            f"network ={node_names2}"
        )
        return
    if len(node_names1) == 0:
        print("[scheduler] skipping cycle: no nodes discovered")
        return

    try:
        scores1 = model.predict(feature_matrix, node_names1)
        scores2 = network_score(node_names1,network_matrix)
        a = 0.4

        scores = {
           k: round((v * (1-a) + a * scores2.get(k,0.0)),4)  for k,v in scores1.items()
        }

    except Exception as e:
        print(f"[scheduler] inference failed: {e}")
        traceback.print_exc()
        return

    cache.set_many(scores)
    print(f"[scheduler] cycle complete — scored {len(scores)} nodes: {scores}")


def _loop():
    print(f"[scheduler] starting, interval={INTERVAL_SECONDS}s")
    while True:
        start = time.monotonic()
        try:
            run_once()
        except Exception as e:
            # catch-all so a bug in _run_once never kills the thread
            print(f"[scheduler] unhandled error in cycle: {e}")
            traceback.print_exc()

        elapsed = time.monotonic() - start
        sleep_for = max(0.0, INTERVAL_SECONDS - elapsed)
        print(f"[scheduler] cycle took {elapsed:.2f}s, sleeping {sleep_for:.2f}s")
        time.sleep(sleep_for)


def start_scheduler():
    """Start the background scoring loop as a daemon thread."""
    t = threading.Thread(target=_loop, name="scoring-loop", daemon=True)
    t.start()
    return t