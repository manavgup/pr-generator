from prometheus_client import Counter, Histogram, Gauge
import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class Telemetry:
    def __init__(self):
        self.logger = structlog.get_logger()
        self.tracer = trace.get_tracer(__name__)
        
        # Metrics
        self.tool_calls = Counter(
            'mcp_tool_calls_total',
            'Total number of MCP tool calls',
            ['tool_name', 'status']
        )
        
        self.tool_duration = Histogram(
            'mcp_tool_duration_seconds',
            'Duration of MCP tool calls',
            ['tool_name']
        )
        
        self.active_sessions = Gauge(
            'mcp_active_sessions',
            'Number of active MCP sessions'
        )
    
    def track_tool_call(self, tool_name: str):
        """Decorator to track tool calls"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(f"tool:{tool_name}") as span:
                    span.set_attribute("tool.name", tool_name)
                    
                    with self.tool_duration.labels(tool_name=tool_name).time():
                        try:
                            result = await func(*args, **kwargs)
                            self.tool_calls.labels(
                                tool_name=tool_name,
                                status="success"
                            ).inc()
                            span.set_status(trace.Status(trace.StatusCode.OK))
                            return result
                        except Exception as e:
                            self.tool_calls.labels(
                                tool_name=tool_name,
                                status="error"
                            ).inc()
                            span.record_exception(e)
                            span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            raise
            return wrapper
        return decorator
