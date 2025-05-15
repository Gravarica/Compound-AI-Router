# src/routing/router_factory.py
from typing import Dict, Any, Optional

from src.routing import BaseRouter
from src.routing import TransformerRouter
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
        if router_type.lower() == 'transformer':
            logger.info(f"Creating TransformerRouter with config: {config}")
            return TransformerRouter(
                model_name_or_path=config['model_name_or_path'],
                num_labels=config.get('num_labels', 2),
                device=config.get('device'),
                max_length=config.get('max_length', 512)
            )
        else:
            raise ValueError(f"Unsupported router type: {router_type}")