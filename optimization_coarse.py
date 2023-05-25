import optuna
import os

os.environ[
    "train_data_path"] = "/cs/labs/tomhope/oziel/runs/seq2rel/coarse_optimization/coarse_transform/train_transform.tsv"
os.environ[
    "valid_data_path"] = "/cs/labs/tomhope/oziel/runs/seq2rel/coarse_optimization/coarse_transform/dev_transform.tsv"
os.environ["dataset_size"] = "673"


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("max_length", 256, 1024)
    trial.suggest_int("max_steps", 80, 120)
    trial.suggest_int("reinit_layers", 1, 3)

    trial.suggest_float("decoder_lr", 3e-4, 6e-4, log=True)
    trial.suggest_float("encoder_lr", 1e-5, 4e-5, log=True)
    trial.suggest_float("dropout", 0.05, 0.3)
    trial.suggest_float("weight_dropout", 0.3, 0.7)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="training_config/coarse_optimize.jsonnet",  # jsonnet path
        serialization_dir=f"./result/optuna/{trial.number}",  # directory for snapshots and logs
        include_package="seq2rel",
        metrics="best_validation_fscore"
    )
    return executor.run()


study = optuna.create_study(
    storage="sqlite:///result/trial.db",  # save results in DB
    sampler=optuna.samplers.TPESampler(seed=24),
    study_name="optuna_allennlp1",
    direction="maximize",
    load_if_exists=True

)

timeout = 60 * 60 * 6  # timeout (sec): 60*60*10 sec => 10 hours

study.optimize(
    objective,
    n_jobs=1,  # number of processes in parallel execution
    n_trials=30,  # number of trials to train a model
    timeout=timeout  # threshold for executing time (sec)
)