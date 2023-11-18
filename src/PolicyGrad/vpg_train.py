from src.utils import get_agent_id
agent_id = get_agent_id(ENV_NAME)

# tensorboard logger to see training curves
from src.utils import get_logger, get_model_path
logger = get_logger(env_name=ENV_NAME, agent_id=agent_id)

# path to save policy network weights and hyperparameters
model_path = get_model_path(env_name=ENV_NAME, agent_id=agent_id)

# let's train!
agent.train(
    n_policy_updates=5000,
    batch_size=256,
    logger=logger,
    model_path=model_path,
)