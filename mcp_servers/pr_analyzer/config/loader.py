# mcp_servers/pr_analyzer/config/loader.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    """Find the project root by looking for specific markers, prioritizing pyproject.toml."""
    current = Path(__file__).resolve()

    # Prioritize pyproject.toml as the primary marker
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent

    # Fallback to other markers if pyproject.toml is not found
    current = Path(__file__).resolve() # Reset current to start searching again
    markers = ['setup.py', '.git', 'requirements.txt'] # Exclude pyproject.toml

    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    # Fallback to 3 levels up from this file if no markers are found
    return Path(__file__).resolve().parents[3]

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def load_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    # Server settings
    if server_name := os.getenv('PR_ANALYZER_SERVER_NAME'):
        config.setdefault('server', {})['name'] = server_name
    
    if server_version := os.getenv('PR_ANALYZER_SERVER_VERSION'):
        config.setdefault('server', {})['version'] = server_version
    
    if server_port := os.getenv('PR_ANALYZER_PORT'):
        config.setdefault('server', {})['port'] = int(server_port)
    
    # Logging settings
    if log_level := os.getenv('PR_ANALYZER_LOG_LEVEL'):
        config.setdefault('logging', {})['level'] = log_level
    
    # Analysis settings
    if max_files := os.getenv('PR_ANALYZER_MAX_FILES_PER_PR'):
        config.setdefault('analysis', {})['max_files_per_pr'] = int(max_files)
    
    if batch_size := os.getenv('PR_ANALYZER_BATCH_SIZE'):
        config.setdefault('analysis', {})['batch_size'] = int(batch_size)
    
    # Security settings
    if security_enabled := os.getenv('PR_ANALYZER_SECURITY_ENABLED'):
        config.setdefault('security', {})['enabled'] = security_enabled.lower() == 'true'
    
    return config

def load_config(config_name: str = 'server.yaml', environment: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from project config directory."""
    project_root = find_project_root()
    config_dir = project_root / 'config'
    
    # Determine environment
    if environment is None:
        environment = os.getenv('PR_ANALYZER_ENV', 'development')
    
    # Load base configuration
    base_config_path = config_dir / config_name
    if not base_config_path.exists():
        logger.warning(f"Config file not found: {base_config_path}")
        return {}
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Load environment-specific overrides
    env_config_path = config_dir / 'environments' / f'{environment}.yaml'
    if env_config_path.exists():
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f) or {}
            config = merge_configs(config, env_config)
    
    # Load additional config files if specified
    additional_configs = [
        ('security', 'security.yaml'),
        ('workflows', 'workflows.yaml'),
        ('mcp_servers', 'mcp_servers.yaml')
    ]
    
    for key, filename in additional_configs:
        config_path = config_dir / filename
        if config_path.exists():
            with open(config_path, 'r') as f:
                additional_config = yaml.safe_load(f) or {}
                config[key] = additional_config
    
    # Apply environment variable overrides
    config = load_env_overrides(config)
    
    return config

def setup_logging(log_config: Dict[str, Any]):
    """Setup logging based on configuration."""
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup basic logging
    logging.basicConfig(level=level, format=format_str)
    
    # Setup file logging if specified
    if log_file := log_config.get('file'):
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)
    
    # Suppress noisy loggers
    for logger_name in log_config.get('suppress', []):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
