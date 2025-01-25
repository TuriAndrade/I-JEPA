from .ijepa_train import main as ijepa_train
from .vit_clf_train import main as vit_clf_train
from .vit_clf_eval import main as vit_clf_eval
from .vit_lidar_eval import main as vit_lidar_eval

module_dict = {
    "ijepa_train": ijepa_train,
    "vit_clf_train": vit_clf_train,
    "vit_clf_eval": vit_clf_eval,
    "vit_lidar_eval": vit_lidar_eval,
}
