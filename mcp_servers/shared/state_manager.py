import redis
import json
from typing import Optional, Dict, Any

class StateManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour default TTL
    
    async def save_analysis(self, session_id: str, analysis: Dict[str, Any]):
        """Save analysis results for a session"""
        key = f"pr_analyzer:analysis:{session_id}"
        self.redis.setex(key, self.ttl, json.dumps(analysis))
    
    async def get_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results for a session"""
        key = f"pr_analyzer:analysis:{session_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    async def save_workflow_state(self, workflow_id: str, state: Dict[str, Any]):
        """Save workflow execution state"""
        key = f"workflow:state:{workflow_id}"
        self.redis.setex(key, self.ttl, json.dumps(state))
