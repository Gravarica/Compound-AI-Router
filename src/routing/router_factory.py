# src/routing/router_factory.py
from typing import Dict, Any, Optional

from src.routing.base_router import BaseRouter
from src.routing.transformer_router import TransformerRouter
from src.routing.random_router import RandomRouter
from src.routing.oracle_router import OracleRouter
from src.utils.logging import setup_logging

logger = setup_logging(name="router_factory")

class RouterFactory:

    @staticmethod
    def create_router(router_type: str, config: Dict[str, Any]) -> BaseRouter:
        """
        Create a router instance.

        Args:
            router_type: Type of router to create
            config: Configuration dictionary

        Returns:
            Router instance

        Raises:
            ValueError: If router_type is not supported
        """
        router_type = router_type.lower()
        if router_type == 'transformer':
            logger.info(f"Creating TransformerRouter with config: {config}")
            return TransformerRouter(
                model_name_or_path=config['model_path'],
                num_labels=config.get('num_labels', 2),
                device=config.get('device'),
                max_length=config.get('max_length', 512)
            )
        elif router_type == 'random':
            logger.info(f"Creating RandomRouter with config: {config}")
            return RandomRouter(
                seed=config.get('seed')
            )
        elif router_type == 'oracle':
            logger.info(f"Creating OracleRouter")
            if 'evaluation_set' not in config:
                raise ValueError("OracleRouter requires 'evaluation_set' in its configuration.")
            return OracleRouter(
                evaluation_set=config['evaluation_set']
            )
        else:
            raise ValueError(f"Unsupported router type: {router_type}")