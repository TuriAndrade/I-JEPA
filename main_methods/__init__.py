from .ijepa_train import main as ijepa_train
from .ijepa_lidar_eval import main as ijepa_lidar_eval
from .ijepa_mine_eval import main as ijepa_mine_eval
from .vit_clf_train import main as vit_clf_train
from .vit_clf_eval import main as vit_clf_eval

module_dict = {
    "ijepa_train": ijepa_train,
    "ijepa_lidar_eval": ijepa_lidar_eval,
    "ijepa_mine_eval": ijepa_mine_eval,
    "vit_clf_train": vit_clf_train,
    "vit_clf_eval": vit_clf_eval,
}
