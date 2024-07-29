import argparse
import torch
from .model import M
import torch
import json
#from torchvision import transforms

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')

from .holistic_from_file import get_landmarks, get_landmarks_from_video

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--work_dir', type=str, default='work_dirs/exp1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

args = parse_args()
# setup_seed(args.seed)

nlayers = args.nlayers
f = open("/Users/emerald.zhang@viam.com/asl-project/kaggleasl5thplacesolution/codesmall/sign_to_prediction_index_map.json")
word_dict = json.load(f)

# ema.restore()

def get_model_output(video):
    model = M(nlayers)
    checkpoint = torch.load('/Users/emerald.zhang@viam.com/asl-project/epoch_399.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    data = torch.from_numpy(get_landmarks_from_video(video))
    data = data.squeeze(1)
    print(type(data), data.shape)
    with torch.no_grad():
        res = model(data)
    print("res: ", res)
    res = res.argmax(dim=-1)
    # y = np.bincount(res, minlength = max(res))
    # y = np.argmax(y)   
    idx = str(res.item())
    return word_dict[idx]

# get_model_output()