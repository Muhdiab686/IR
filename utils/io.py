import joblib
import json
from pathlib import Path

class PersistenceManager:
    """إدارة عمليات التحميل والحفظ لكل الملفات والنماذج (joblib, json)."""

    @staticmethod
    def save_joblib(obj, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        print(f"✅ Saved joblib to {path}")

    @staticmethod
    def load_joblib(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return joblib.load(path)
    
    @staticmethod
    def save_json(obj, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
        print(f"✅ Saved JSON to {path}")

    @staticmethod
    def load_json(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
