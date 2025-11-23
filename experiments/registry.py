from algorithms.standalone.ppo import PPO
from algorithms.standalone.es import ES
from algorithms.hybrid.es_ppo import ESPPO
from algorithms.hybrid.max_ppo import MaxPPO
from algorithms.hybrid.alt_ppo import AltPPO

ALGORITHM_REGISTRY = {
    'ppo': PPO,
    'es': ES,
    'esppo': ESPPO,
    'maxppo': MaxPPO,
    'altppo': AltPPO,
}

def get_algorithm(name: str):
    """Get algorithm class by name."""
    name_lower = name.lower()
    if name_lower not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name_lower]

