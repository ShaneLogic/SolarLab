from celery import Celery

celery_app = Celery(
    'perovskite_sim_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def run_jv_task(config_path):
    from perovskite_sim.experiments import jv_sweep
    return jv_sweep.run_jv_sweep(config_path)

@celery_app.task
def run_impedance_task(config_path):
    from perovskite_sim.experiments import impedance
    return impedance.run_impedance(config_path)

@celery_app.task
def run_degradation_task(config_path):
    from perovskite_sim.experiments import degradation
    return degradation.run_degradation(config_path)
