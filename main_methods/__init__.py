from .ijepa_train import main as ijepa_train
from .ijepa_ddp_train import main as ijepa_ddp_train
from .vit_clf_train import main as vit_clf_train
from .vit_clf_ddp_train import main as vit_clf_ddp_train
from .vit_clf_eval import main as vit_clf_eval

module_dict = {
    "ijepa_train": ijepa_train,
    "ijepa_ddp_train": ijepa_ddp_train,
    "vit_clf_train": vit_clf_train,
    "vit_clf_ddp_train": vit_clf_ddp_train,
    "vit_clf_eval": vit_clf_eval,
}
