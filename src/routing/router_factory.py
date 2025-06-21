# src/routing/router_factory.py
from typing import Dict, Any, Optional

from src.routing.base_router import BaseRouter
from src.routing.transformer_router import TransformerRouter
from src.routing.random_router import RandomRouter
from src.routing.oracle_router import OracleRouter
from src.routing.llm_router import LLMRouter
from src.routing.cot_llm_router import ChainOfThoughtLLMRouter
from src.models import LLMFactory
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
                model_name_or_path=config.get('model_path', config.get('model_name_or_path')),
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
        elif router_type == 'llm':
            logger.info(f"Creating LLMRouter with config: {config}")
            if 'llm_config' not in config:
                raise ValueError("LLMRouter requires 'llm_config' in its configuration.")
            
            llm_config = config['llm_config']
            llm = LLMFactory.create_llm(llm_config['type'], llm_config)
            
            return LLMRouter(
                llm=llm,
                confidence_threshold=config.get('confidence_threshold', 0.7)
            )
        elif router_type == 'cot_llm':
            logger.info(f"Creating ChainOfThoughtLLMRouter with config: {config}")
            if 'llm_config' not in config:
                raise ValueError("ChainOfThoughtLLMRouter requires 'llm_config' in its configuration.")
            
            llm_config = config['llm_config']
            llm = LLMFactory.create_llm(llm_config['type'], llm_config)
            
            return ChainOfThoughtLLMRouter(
                llm=llm,
                confidence_threshold=config.get('confidence_threshold', 0.7)
            )
        else:
            raise ValueError(f"Unsupported router type: {router_type}")