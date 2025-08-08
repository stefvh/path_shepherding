from . import metric, metric_only_repulsive

model_dir = {
    'METRIC': metric.MetricModel,
    'METRIC-ONLY-REPULSIVE': metric_only_repulsive.MetricOnlyRepulsive,
}

def get_model_names():
    return list(model_dir.keys())

def get_model(model_name, **kwargs):
    return model_dir[model_name](**kwargs)