import pickle
import torch
import numpy as np
import argparse
from dataset import make_dataset
from networks import make_model
from tqdm import tqdm
from utils import init_weights, calc_accuracy

parser = argparse.ArgumentParser("test")
parser.add_argument('--model_type', type=str, help='Write the model name in the conv_g_f format. For example, patch_light_heavy', required=True)
parser.add_argument('--model', type=str, help='Path of pretrained model.', required=True)
parser.add_argument('--dataset', type=str, help='Path of Sort-of-CLEVR dataset', required=True)
config, _ = parser.parse_known_args()

batch_size = 64
resume = config.model
model_dict = dict(zip('conv_g_f'.split('_'), config.model_type.split('_')))

np_dataset = pickle.load(open(config.dataset, 'rb'))
_, np_val_dataset = np_dataset
val_rel_dataset, _ = make_dataset(np_val_dataset)
val_dataloader = torch.utils.data.DataLoader(val_rel_dataset, batch_size=batch_size,
                                             pin_memory=True, drop_last=False)

net = make_model(model_dict).cuda()

print('load model from {}'.format(resume))
prev = torch.load(resume)
net.load_state_dict(prev['net'])

net.eval()
accuracies = []

with torch.no_grad():
    for i, (image, question, answer) in tqdm(enumerate(val_dataloader), desc='val'):
        image, question, answer = image.cuda(), question.cuda(), answer.cuda()
        pred = net(image, question)
        accuracy = calc_accuracy(pred, answer)
        accuracies += [accuracy] * answer.size(0)

val_accuracy = sum(accuracies) / len(accuracies)
print(val_accuracy)