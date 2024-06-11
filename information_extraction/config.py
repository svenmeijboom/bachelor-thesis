from pathlib import Path

WANDB_PROJECT = 'MovieName-extraction'
#WANDB_PROJECT = 'swde-extraction'
#WANDB_PROJECT = 'bachelor-thesis'
#DATA_DIR = Path('~/Data/').expanduser()
#str_path = "/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data/MovieName-set/"
#str_path = "/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data2/"
str_path = "/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data/"
DATA_DIR = Path(str_path)

#DEFAULT_GROUND_TRUTH_DIR_CONF = DATA_DIR / 'swde-set200' / 'groundtruth'
#DEFAULT_GROUND_TRUTH_DIR_CONF = DATA_DIR / 'swde-set2000' / 'groundtruth'
DEFAULT_GROUND_TRUTH_DIR_CONF = DATA_DIR / 'swde-set200' / 'groundtruth'