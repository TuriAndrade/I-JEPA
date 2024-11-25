from .ijepa_train import main as ijepa_train
from .ijepa_mine_train import main as ijepa_mine_train
from .ijepa_mine_mse_train import main as ijepa_mine_mse_train
from .vit_clf_train import main as vit_clf_train
from .vit_clf_eval import main as vit_clf_eval

module_dict = {
    "ijepa_train": ijepa_train,
    "ijepa_mine_train": ijepa_mine_train,
    "ijepa_mine_mse_train": ijepa_mine_mse_train,
    "vit_clf_train": vit_clf_train,
    "vit_clf_eval": vit_clf_eval,
}
