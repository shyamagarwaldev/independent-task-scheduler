import threading
from  datetime import timezone, datetime
from typing import Dict

# tread safe cache

class ScoreCache:
    def __init__(self):
        self._lock = threading.RLock()
        self._score: Dict[str,Dict] = {}

    def set(self,node_name:str, score:float):
        with self._lock:
            self._score[node_name] = {
                "score": round(score,4),
                "node_name": node_name,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
    def set_many(self, scores: Dict[str,float]):
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            for node_name, score in scores.items():
                self._score[node_name] = {
                "score": round(score,4),
                "node_name": node_name,
                "last_updated": now
            }
    def get_all(self) -> Dict[str,Dict]:
        with self._lock:
            return dict(self._score)

    def get(self,node_name:str) -> dict | None:
        with self._lock:
            if node_name not in self._score:
                return None
            return self._score.get(node_name)

cache = ScoreCache()   
