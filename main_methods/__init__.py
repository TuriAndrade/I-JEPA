from .ijepa_train import main as ijepa_train
from .ijepa_ddp_train import main as ijepa_ddp_train

module_dict = {
    "ijepa_train": ijepa_train,
    "ijepa_ddp_train": ijepa_ddp_train,
}
