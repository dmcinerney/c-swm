import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import logging
from train_sim import make_pairwise_encoder

from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.utils.data import random_split

import modules

from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')
parser.add_argument('--split-mlp', action='store_true', default=False,
                    help='Create two MLPs, one for movable and the other for immovable objects.')
parser.add_argument('--split-gnn', action='store_true', default=False)
parser.add_argument('--immovable-bit', action='store_true', default=False)
parser.add_argument('--same-ep-neg', action='store_true', default=False)
parser.add_argument('--only-same-ep-neg', action='store_true', default=False)
parser.add_argument('--no-loss-first-two', action='store_true', default=False)
parser.add_argument('--bisim', action='store_true', default=False)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--custom-neg', default=False, action='store_true')
parser.add_argument('--bisim-metric-path')
parser.add_argument('--bisim-eps', type=float)
parser.add_argument('--bisim-model-path')
parser.add_argument('--next-state-neg', default=False, action="store_true")
parser.add_argument('--nl-type', default=0, type=int)

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')
parser.add_argument('--gess', action='store_true', default=False,
                    help='Train model using gess objective instead of pixel-based loss. Must be used with decoder.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--eval_dataset', type=str,
                    default='data/shapes_val.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')
loss_file = os.path.join(save_folder, 'loss.pdf')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

if args.bisim:
    if args.custom_neg:
        dataset = utils.StateTransitionsDatasetStateIdsNegs(
            hdf5_file=args.dataset)
        if args.eval_dataset is not None:
            eval_dataset = utils.StateTransitionsDatasetStateIdsNegs(
                hdf5_file=args.eval_dataset)
    else:
        dataset = utils.StateTransitionsDatasetStateIds(
            hdf5_file=args.dataset)
        if args.eval_dataset is not None:
            eval_dataset = utils.StateTransitionsDatasetStateIds(
                hdf5_file=args.eval_dataset)
else:
    assert not args.custom_neg
    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset)
    if args.eval_dataset is not None:
        eval_dataset = utils.StateTransitionsDataset(
            hdf5_file=args.eval_dataset)

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.eval_dataset is not None:
    subset = 10
    total = len(eval_dataset)
    eval_subset = random_split(eval_dataset, [subset, total-subset])[0]
    eval_loader = data.DataLoader(
        eval_subset, batch_size=args.batch_size)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

# maybe load bisim metric and turn it into torch tensor on the selected device
bisim_metric = None
if args.bisim_metric_path is not None:
    bisim_metric = torch.tensor(np.load(args.bisim_metric_path), dtype=torch.float32, device=device)

# maybe load bisim model
bisim_model = None
if args.bisim_model_path is not None:
    bisim_model = make_pairwise_encoder()
    bisim_model.load_state_dict(torch.load(args.bisim_model_path))

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    split_mlp=args.split_mlp,
    same_ep_neg=args.same_ep_neg,
    only_same_ep_neg=args.only_same_ep_neg,
    immovable_bit=args.immovable_bit,
    split_gnn=args.split_gnn,
    no_loss_first_two=args.no_loss_first_two,
    gamma=args.gamma,
    bisim_metric=bisim_metric,
    bisim_eps=args.bisim_eps,
    next_state_neg=args.next_state_neg,
    nl_type=args.nl_type,
    encoder=args.encoder).to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

# avoid re-initializing the model and adding to the trainable parameters list
if bisim_model is not None:
    model.bisim_model = bisim_model
    model.bisim_model.to(device)

if args.decoder:
    if args.encoder == 'large':
        decoder = modules.DecoderCNNLarge(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'medium':
        decoder = modules.DecoderCNNMedium(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'small':
        decoder = modules.DecoderCNNSmall(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    decoder.apply(utils.weights_init)
    optimizer_dec = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)


class Scheduler:
    def get_value(self, step):
        raise NotImplementedError

class ExpSinScheduler(Scheduler):
    def __init__(self, coefficient, period, offset, base):
        self.coefficient = coefficient
        self.period = period
        self.offset = offset
        self.base = base

    def get_value(self, step):
        return self.base ** (self.offset + self.coefficient * np.sin(2 * np.pi * step / self.period))

class StepScheduler:
    def __init__(self, values, steps):
        assert len(values) == len(steps) + 1
        self.values = values
        self.steps = steps

    def get_value(self, step):
        for i in range(len(self.steps)):
            if step < sum(self.steps[:i+1]):
                return self.values[i]
        return self.values[-1]

# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
losses = []
logger = SummaryWriter()
log_examples = [dataset[i] for i in range(30)]
if args.eval_dataset is not None:
    eval_log_examples = [eval_dataset[i] for i in range(10)]
# sigma = ExpSinScheduler(.001, 2000, 0, 10)
# sigma = ExpSinScheduler(0, 2000, 1, args.sigma)
# sigma = StepScheduler([1, .9, .8, .7, .6, .5], [5000, 5000, 5000, 5000, 5000])
sigma = StepScheduler([args.sigma], [])
freeze_encoder = StepScheduler([False], [])
freeze_decoder = StepScheduler([False], [])
freeze_predictor = StepScheduler([False], [])

def plot_states(states, next_states, next_pred_states):
    figure = plt.figure(figsize=(10, 10))
    i, o, c = states.shape
    allstates = np.concatenate([states.reshape(i*o, 1, c),
                                next_states.reshape(i*o, 1, c),
                                next_pred_states.reshape(i*o, 1, c)], 1)
    allstates2 = TSNE(n_components=2).fit_transform(allstates.reshape(i*o*3, c)).reshape(i*o, 3, 2) \
                 if c > 2 else allstates
    plt.plot(allstates2[:, 0, 0], allstates2[:, 0, 1], '.')
    plt.plot(allstates2[:, 1, 0], allstates2[:, 1, 1], 'g.')
    plt.plot(allstates2[:, 2, 0], allstates2[:, 2, 1], 'r.')
    for inst in range(allstates2.shape[0]):
        plt.arrow(allstates2[inst, 0, 0], allstates2[inst, 0, 1],
                  allstates2[inst, 1, 0] - allstates2[inst, 0, 0],
                  allstates2[inst, 1, 1] - allstates2[inst, 0, 1], color='g')
        plt.arrow(allstates2[inst, 0, 0], allstates2[inst, 0, 1],
                  allstates2[inst, 2, 0] - allstates2[inst, 0, 0],
                  allstates2[inst, 2, 1] - allstates2[inst, 0, 1], color='r')
    return figure

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        if step == 0 or freeze_encoder.get_value(step) != freeze_encoder.get_value(step-1):
            for p in model.obj_encoder.parameters():
                p.requires_grad_(not freeze_encoder.get_value(step))
            for p in model.obj_extractor.parameters():
                p.requires_grad_(not freeze_encoder.get_value(step))
        if args.decoder:
            if step == 0 or freeze_decoder.get_value(step) != freeze_decoder.get_value(step-1):
                for p in decoder.parameters():
                    p.requires_grad_(not freeze_decoder.get_value(step))
        if step == 0 or freeze_predictor.get_value(step) != freeze_predictor.get_value(step-1):
            for p in model.transition_model.parameters():
                p.requires_grad_(not freeze_predictor.get_value(step))

        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        if args.decoder:
            assert not args.bisim # not implemented
            optimizer_dec.zero_grad()
            obs, action, next_obs = data_batch
            objs = model.obj_extractor(obs)
            state = model.obj_encoder(objs)

            rec = torch.sigmoid(decoder(state))

            next_state_pred = state + model.transition_model(state, action)

            if args.gess:
                model.sigma = sigma.get_value(step)
                next_objs = model.obj_extractor(next_obs)
                next_state = model.obj_encoder(next_objs)
                next_state_pert = Normal(next_state, torch.ones_like(next_state)).rsample()
                next_rec = torch.sigmoid(decoder(next_state_pert))
                rec_loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)
                next_rec_loss = F.binary_cross_entropy(
                    next_rec, next_obs,
                    reduction='sum') / obs.size(0)
                loss = (rec_loss + next_rec_loss) / 2
                # next_rec_pred = torch.sigmoid(decoder(next_state_pred.detach()))
                # next_rec_pred_loss = F.binary_cross_entropy(
                #     next_rec_pred, next_obs,
                #     reduction='sum') / obs.size(0)
                # loss = (rec_loss + next_rec_loss + next_rec_pred_loss) / 3
                logger.add_scalar('reco_loss', loss.item(), global_step=step)
                next_state_loss = model.energy(state, action, next_state).sum() / obs.size(0)
                logger.add_scalar('latent_loss', next_state_loss.item(), global_step=step)
                loss += next_state_loss
            else:
                next_rec_pred = torch.sigmoid(decoder(next_state_pred))
                loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)
                next_loss = F.binary_cross_entropy(
                    next_rec_pred, next_obs,
                    reduction='sum') / obs.size(0)
                loss += next_loss / 2
                next_objs = model.obj_extractor(next_obs)
                next_state = model.obj_encoder(next_objs)
                next_state_loss = model.energy(state, action, next_state).sum() / obs.size(0)
                logger.add_scalar('reco_loss', loss.item(), global_step=step)
                logger.add_scalar('latent_loss', next_state_loss.item(), global_step=step)
        else:
            loss = model.contrastive_loss(*data_batch)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        losses.append(loss.item())

        if args.decoder:
            optimizer_dec.step()

        model.eval()
        with torch.no_grad():
            # logging
            logger.add_scalar('loss', loss.item(), global_step=step)
            if step % 100 == 0:
                if args.decoder:
                    for i,example in enumerate(log_examples):
                        obs, action, next_obs = example
                        obs = torch.tensor(obs, device=device).unsqueeze(0)
                        action = torch.tensor(action, device=device).unsqueeze(0)
                        next_obs = torch.tensor(next_obs, device=device).unsqueeze(0)
                        objs = model.obj_extractor(obs)
                        state = model.obj_encoder(objs)
                        rec = torch.sigmoid(decoder(state))
                        next_objs = model.obj_extractor(next_obs)
                        next_state = model.obj_encoder(next_objs)
                        next_rec = torch.sigmoid(decoder(next_state))
                        next_state_pred = state + model.transition_model(state, action)
                        next_rec_pred = torch.sigmoid(decoder(next_state_pred))
                        x = torch.cat([obs, rec, next_obs, next_rec, next_rec_pred], 3).squeeze(0)
                        # masks = torch.cat([objs[0, i] for i in range(objs.shape[1])], 1)
                        # masks = torch.nn.functional.upsample(masks.unsqueeze(0).unsqueeze(0),
                        #                              (x.shape[1], x.shape[1]*objs.shape[1])).squeeze(0).expand(
                        #     x.shape[0], x.shape[1], x.shape[1]*objs.shape[1])
                        # x = torch.nn.functional.pad(x, [0, max(0, masks.shape[2] - x.shape[2])])
                        # masks = torch.nn.functional.pad(masks, [0, max(0, x.shape[2] - masks.shape[2])])
                        # x = torch.cat([x, masks], 1)
                        figure = plt.figure(figsize=(5 * x.shape[2] // x.shape[1], 5))
                        plt.subplot(1, 1, 1, title='prefix')
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(utils.css_to_ssc(utils.to_np(x[:3])))
                        logger.add_figure('train/example'+str(i), figure, global_step=step)
                    if args.eval_dataset is not None:
                        for i,example in enumerate(eval_log_examples):
                            obs, action, next_obs = example
                            obs = torch.tensor(obs, device=device).unsqueeze(0)
                            action = torch.tensor(action, device=device).unsqueeze(0)
                            next_obs = torch.tensor(next_obs, device=device).unsqueeze(0)
                            objs = model.obj_extractor(obs)
                            state = model.obj_encoder(objs)
                            rec = torch.sigmoid(decoder(state))
                            next_objs = model.obj_extractor(next_obs)
                            next_state = model.obj_encoder(next_objs)
                            next_rec = torch.sigmoid(decoder(next_state))
                            next_state_pred = state + model.transition_model(state, action)
                            next_rec_pred = torch.sigmoid(decoder(next_state_pred))
                            x = torch.cat([obs, rec, next_obs, next_rec, next_rec_pred], 3).squeeze(0)
                            # masks = torch.cat([objs[0, i] for i in range(objs.shape[1])], 1)
                            # masks = torch.nn.functional.upsample(masks.unsqueeze(0).unsqueeze(0),
                            #                                      (x.shape[1], x.shape[1] * objs.shape[1])).squeeze(
                            #     0).expand(
                            #     x.shape[0], x.shape[1], x.shape[1] * objs.shape[1])
                            # x = torch.nn.functional.pad(x, [0, max(0, masks.shape[2] - x.shape[2])])
                            # masks = torch.nn.functional.pad(masks, [0, max(0, x.shape[2] - masks.shape[2])])
                            # x = torch.cat([x, masks], 1)
                            figure = plt.figure(figsize=(5 * x.shape[2] // x.shape[1], 5))
                            plt.subplot(1, 1, 1, title='prefix')
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            plt.imshow(utils.css_to_ssc(utils.to_np(x[:3])))
                            logger.add_figure('eval/example'+str(i), figure, global_step=step)
                for i, example in enumerate(log_examples):
                    obs, action, next_obs = example
                    obs = torch.tensor(obs, device=device).unsqueeze(0)
                    objs = model.obj_extractor(obs)
                    obs = obs.squeeze(0)
                    masks = torch.cat([objs[0, i] for i in range(objs.shape[1])], 1)
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0).unsqueeze(0), (obs.shape[1], obs.shape[1] * objs.shape[1])).squeeze(
                        0).expand(obs.shape[0], obs.shape[1], obs.shape[1] * objs.shape[1])
                    x = torch.cat([obs, masks], 2)
                    figure = plt.figure(figsize=(5 * (objs.shape[1] + 1), 5))
                    plt.subplot(1, 1, 1, title='prefix')
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(utils.css_to_ssc(utils.to_np(x[:3])))
                    logger.add_figure('train/example' + str(i) + '_masks', figure, global_step=step)
                if args.eval_dataset is not None:
                    for i, example in enumerate(eval_log_examples):
                        obs, action, next_obs = example
                        obs = torch.tensor(obs, device=device).unsqueeze(0)
                        objs = model.obj_extractor(obs)
                        obs = obs.squeeze(0)
                        masks = torch.cat([objs[0, i] for i in range(objs.shape[1])], 1)
                        masks = torch.nn.functional.interpolate(
                            masks.unsqueeze(0).unsqueeze(0), (obs.shape[1], obs.shape[1] * objs.shape[1])).squeeze(
                            0).expand(obs.shape[0], obs.shape[1], obs.shape[1] * objs.shape[1])
                        x = torch.cat([obs, masks], 2)
                        figure = plt.figure(figsize=(5 * (objs.shape[1] + 1), 5))
                        plt.subplot(1, 1, 1, title='prefix')
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(utils.css_to_ssc(utils.to_np(x[:3])))
                        logger.add_figure('eval/example' + str(i) + '_masks', figure, global_step=step)
                    states = []
                    next_states = []
                    next_pred_states = []
                    for batch_idx, data_batch in enumerate(eval_loader):
                        data_batch = [tensor.to(device) for tensor in data_batch]
                        obs, action, next_obs = data_batch
                        objs = model.obj_extractor(obs)
                        state = model.obj_encoder(objs)
                        next_objs = model.obj_extractor(next_obs)
                        next_state = model.obj_encoder(next_objs)
                        next_state_pred = state + model.transition_model(state, action)
                        states.append(state)
                        next_states.append(next_state)
                        next_pred_states.append(next_state_pred)
                    figure = plot_states(torch.cat(states, 0).detach().cpu().numpy(),
                                         torch.cat(next_states, 0).detach().cpu().numpy(),
                                         torch.cat(next_pred_states, 0).detach().cpu().numpy())
                    logger.add_figure('latent', figure, global_step=step)
        model.train()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_file)

plt.subplot(2, 1, 1)
plt.plot(losses)
plt.subplot(2, 1, 2)
plt.plot(losses)
plt.yscale("log")
plt.savefig(loss_file)
