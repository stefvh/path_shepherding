
from models.base import BaseModel
from models.metric import MetricModel


def get_model(name, **kwargs) -> BaseModel:
    models = {
        'METRIC': MetricModel,
    }

    if name not in models:
        raise ValueError('Invalid model name: {}'.format(name))
    return models[name](**kwargs)
