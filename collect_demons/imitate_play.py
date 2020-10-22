# TODO : *IMPORTANT* - change the imitation policy to be condtioned on the GROUND STATE rather than visual obv
#        While running this imitated policy, collect visual_obv w/ domain randomization

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm # TODO : Remove TQDMs
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from demons_config import get_demons_args

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datatset_env'))

from models import PerceptionModule
from model_config import get_model_args

from file_storage import store_trajectoy

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

class ImitationPolicy(torch.nn.Module):
    def __init__(self, action_dim=8, state_dim=72, hidden_size=2048, batch_size=1, rnn_type='RNN', num_layers=2):
        super(ImitationPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.upper()
        self.num_layers = num_layers

        assert self.rnn_type in ['LSTM', 'GRU', 'RNN']
        self.rnn = {'LSTM' : torch.nn.LSTMCell, 'GRU' : torch.nn.GRUCell, 'RNN' : torch.nn.RNNCell}[self.rnn_type]

        self.rnn1 = self.rnn(self.state_dim, self.hidden_size)
        self.rnn2 = self.rnn(self.hidden_size, self.hidden_size)

        # NOTE : Original paper used a Mixture of Logistic (MoL) dist. Implement later.
        self.hidden2mean = torch.nn.Linear(self.hidden_size, self.action_dim)
        log_std = -0.5 * np.ones(self.action_dim, dtype=np.np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # The hidden states of the RNN.
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def _prepare_obs(self, state, perception_module):
        if state.size()[-1] != self.state_dim:
            assert perception_module is not None
            state = perception_module(state)

        # The hidden states of the RNN.
        self.h1 = torch.randn(state.shape[0], self.hidden_size)
        self.h2 = torch.randn(state.shape[0], self.hidden_size)
        if self.rnn_type == 'LSTM':
            self.c1 = torch.randn(state.shape[0], self.hidden_size)
            self.c2 = torch.randn(state.shape[0], self.hidden_size)

        return state

    def _distribution(self, obs):
        if self.rnn_type == 'LSTM':
            self.h1, self.c1 = self.relu(self.rnn1(obs, (self.h1, self.c1))) 
            self.h2, self.c2 = self.relu(self.rnn2(self.h1, (self.h2, self.c2)))
        else:
            self.h1 = self.relu(self.rnn1(obs, self.h1)) 
            self.h2 = self.relu(self.rnn2(self.h1, self.h2))

        mean = self.tanh(self.hidden2mean(self.h2))
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1) 

    def forward(self, state, action=None, perception_module=None):
        obs = self._prepare_obs(state, perception_module)
        policy = self._distribution(obs)
        
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a

    def step(self, state, perception_module=None):
        with torch.no_grad():
            obs = self._prepare_obs(state, perception_module)
            policy = self._distribution(obs)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)

        return action.numpy(), logp_a.numpy()

def train_imitation(demons_config):
    model_config = get_model_args()

    deg = demons_config.deg(get_episode_type='EPISODE_ROBOT_PLAY')

    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    tensorboard_writer = SummaryWriter(logdir=demons_config.tensorboard_path)
    perception_module = PerceptionModule(vobs_dim, dof_dim, model_config.visual_state_dim).to(device)
    imitation_policy = ImitationPolicy(act_dim, model_config.combined_state_dim,).to(device)

    params = list(perception_module.parameters()) + list(imitation_policy.parameters())
    print("Number of parameters : {}".format(len(params)))

    optimizer = torch.optim.Adam(params, lr=model_config.learning_rate)

    if(demons_config.load_models):
        # TODO : IMPORTANT - Check if file exist before loading
        if demons_config.use_model_perception:
            perception_module.load_state_dict(torch.load(os.path.join(model_config.models_save_path, 'perception.pth')))
        else :
            perception_module.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'perception.pth')))
        imitation_policy.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'imitation_policy.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'optimizer.pth')))

    print("Run : tensorboard --logdir={} --host '0.0.0.0' --port 6006".format(demons_config.tensorboard_path))
    data_loader = DataLoader(deg.traj_dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=1)
    max_step_size = len(data_loader.dataset)

    for epoch in tqdm(range(model_config.max_epochs), desc="Check Tensorboard"):
        for i, trajectory in enumerate(data_loader):
            trajectory = {key : trajectory[key].float().to(device) for key in trajectory.keys()}
            visual_obvs, dof_obs, action = trajectory[deg.vis_obv_key], trajectory[deg.dof_obv_key], trajectory['action']
            batch_size, seq_len =  visual_obvs.shape[0], visual_obvs.shape[1] 

            visual_obvs = visual_obvs.reshape(batch_size * seq_len, vobs_dim[2], vobs_dim[0], vobs_dim[1])
            dof_obs = dof_obs.reshape(batch_size * seq_len, dof_dim)
            actions = trajectory['action'].reshape(batch_size * seq_len, -1)

            states = perception_module(visual_obvs, dof_obs) # DEBUG : Might raise in-place errors

            pi, logp_a = imitation_policy(state=states, action=actions)

            optimizer.zero_grad()
            loss = -logp_a
            loss = loss.mean()

            tensorboard_writer.add_scalar('Clone Loss', loss, epoch * max_step_size + i)
            loss.backward()
            optimizer.step()

            if int(i % model_config.save_interval) == 0:
                if not demons_config.use_model_perception:
                    torch.save(perception_module.state_dict(), os.path.join(demons_config.models_save_path, 'perception.pth'))
                torch.save(imitation_policy.state_dict(), os.path.join(demons_config.models_save_path, 'imitation_policy.pth'))
                torch.save(optimizer.state_dict(), os.path.join(demons_config.models_save_path, 'optimizer.pth'))


def imitate_play():
    model_config = get_model_args()
    demons_config = get_demons_args()

    deg = demons_config.deg()
    env = deg.get_env()
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    with torch.no_grad():
        perception_module = PerceptionModule(vobs_dim, dof_dim, model_config.visual_state_dim).to(device)
        imitation_policy = ImitationPolicy(act_dim, model_config.combined_state_dim,).to(device)
        
        if demons_config.use_model_perception:
            perception_module.load_state_dict(torch.load(os.path.join(model_config.models_save_path, 'perception.pth')))
        else :
            perception_module.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'perception.pth')))
        imitation_policy.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'imitation_policy.pth')))

        for run in range(demons_config.n_gen_traj):
            obs = env.reset()
            tr_vobvs, tr_dof, tr_actions = [], [], []

            for step in range(demon_config.flush_freq):
                visual_obv, dof_obv = torch.from_numpy(obvs[deg.vis_obv_key]).float(), torch.from_numpy(obvs[deg.dof_obv_key]).float()
                visual_obv = visual_obv.reshape(1, visual_obv.shape[2], visual_obv.shape[0], visual_obv.shape[1])
                dof_obv = dof_obv.reshape(1, dof_obv.shape[0])
                state = perception_module(visual_obv, dof_obv)

                action, _ = imitation_policy.step(state)
                
                if int(step % demons_config.collect_freq) == 0:
                    tr_vobvs.append(visual_obv)
                    tr_dof.append(dof_obv)
                    tr_actions.append(action[0])

                obs, _, done, _ = env.step(action[0])

            print('Storing Trajectory')
            trajectory = {deg.vis_obv_key : np.array(tr_vobvs), deg.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
            store_trajectoy(trajectory, 'imitation')
            trajectory, tr_vobvs, tr_dof, tr_actions = {}, [], [], []

        env.close()

if __name__ == '__main__':
    demons_config = get_demons_args()
    if demons_config.train_imitation:
        train_imitation(demons_config)

    imitate_play()