from collections import OrderedDict

from catalyst.utils import set_global_seed
from catalyst.runners import SupervisedRunner
from catalyst import dl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from clearml import Task, Logger

from src.dataset import get_dataloaders
from src.config import Config
from src.model import Model
from src.const import INPUTS, TARGETS, LOGITS, TRAIN, VALID, LOSS

if __name__ == '__main__':
    task = Task.init(
        project_name='test_project',
        task_name='remote_execution nlp stepik 4_3',
    )

    task.execute_remotely(queue_name="default")

    set_global_seed(25)
    config = Config()
    train_dataloader, valid_dataloader = get_dataloaders(config)

    model = Model(config)

    runner = SupervisedRunner(
        model=model,
        input_key=INPUTS,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    callbacks = [
        dl.CriterionCallback(
            input_key=LOGITS,
            target_key=TARGETS,
            metric_key=LOSS,
        ),
        dl.OptimizerCallback(
            metric_key=LOSS,
        ),
        dl.CheckpointCallback(
            logdir='checkpoints',
            loader_key=VALID,
            minimize=True,
            metric_key=LOSS,
            mode='model',
        )
    ]

    runner.train(
        loaders=OrderedDict({TRAIN: train_dataloader, VALID: valid_dataloader}),
        model=model,
        criterion=CrossEntropyLoss(),
        optimizer=Adam(lr=config.lr, params=model.parameters()),
        callbacks=callbacks,
        seed=config.seed,
        num_epochs=config.epochs,
        valid_metric=LOSS,
        valid_loader=VALID,
        minimize_valid_metric=True,
        verbose=True,
        check=False,
    )
