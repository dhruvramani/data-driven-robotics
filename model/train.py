import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm # TODO : Remove TQDM
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import *

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train(config):
    deg = config.deg()
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
    visual_goal_encoder = VisualGoalEncoder(config.visual_state_dim, config.goal_dim).to(device)
    plan_recognizer = PlanRecognizerModule(config.combined_state_dim, act_dim, config.latent_dim).to(device)
    plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
    control_policy = ControlPolicy(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)

    params = list(perception_module.parameters()) + list(visual_goal_encoder.parameters())
    params += list(plan_recognizer.parameters()) + list(plan_proposer.parameters()) + list(control_policy.parameters()) 

    print("Number of parameters : {}".format(len(params)))
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    if(config.save_graphs):
        tensorboard_writer.add_graph(perception_module)
        tensorboard_writer.add_graph(visual_goal_encoder)
        tensorboard_writer.add_graph(plan_recognizer)
        tensorboard_writer.add_graph(plan_proposer)
        tensorboard_writer.add_graph(control_policy)

    if(config.resume):
        # TODO : IMPORTANT - Check if file exist before loading
        # TODO : Implement load & save functions within the class for easier loading and saving
        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_recognizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_recognizer.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_policy.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_policy.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'optimizer.pth')))

    print("Run : `tensorboard --logdir={} --host '0.0.0.0' --port 6006`".format(config.tensorboard_path))
    dataloader = deg.get_traj_dataloader(batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    def inference(trajectory, goal_state):
        visual_obvs, dof_obs, action = trajectory[deg.vis_obv_key], trajectory[deg.dof_obv_key], trajectory['action']
        batch_size, seq_len =  visual_obvs.shape[0], visual_obvs.shape[1] 

        visual_obvs = visual_obvs.reshape(batch_size * seq_len, vobs_dim[2], vobs_dim[0], vobs_dim[1])
        dof_obs = dof_obs.reshape(batch_size * seq_len, dof_dim)
        actions = trajectory['action'].reshape(batch_size * seq_len, -1)

        states = perception_module(visual_obvs, dof_obs) # DEBUG : Might raise in-place errors
        states = states.reshape(batch_size, seq_len, config.combined_state_dim)
        inital_state = states[:, 0]

        prior_z, prior_z_dist, _, _ = plan_proposer(inital_state, goal_state)
        post_z, post_z_dist, _, _ = plan_recognizer(states, action)

        # NOTE : goal_state dims - [batch_size, goal_dim]
        goal_state = goal_state.unsqueeze(1)
        goal_states = goal_state.repeat(1, seq_len, 1)
        post_zs = post_z.unsqueeze(1)
        post_zs = post_zs.repeat(1, seq_len, 1)

        goal_states = goal_states.reshape(batch_size * seq_len, -1)
        post_zs = post_zs.reshape(batch_size * seq_len, -1)
        states = states.reshape(batch_size * seq_len, -1)

        pi, logp_a = control_policy(state=states, goal=goal_states, 
                        zp=post_zs, action=actions)

        loss_pi = -logp_a.mean()
        loss_kl = torch.distributions.kl_divergence(post_z_dist, prior_z_dist).mean()
        loss = loss_pi + config.beta * loss_kl
        return loss

    for epoch in tqdm(range(config.max_epochs), desc="Check Tensorboard"):        
        # NOTE : God bless detect_anomaly() 🙏🙏
        #with torch.autograd.detect_anomaly():
        for i, trajectory in enumerate(dataloader):
            trajectory = {key : trajectory[key].float().to(device) for key in trajectory.keys()}
            
            last_obvs = trajectory[deg.vis_obv_key][:, -1].reshape(trajectory[deg.vis_obv_key].shape[0], vobs_dim[2], vobs_dim[0], vobs_dim[1])
            goal_state = perception_module(last_obvs)
            goal_state, _, _, _ = visual_goal_encoder(goal_state)

            loss = inference(trajectory, goal_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tensorboard_writer.add_scalar('loss/total', loss, epoch * len(dataloader.dataset) + i)

        if int(epoch % config.save_interval_epoch) == 0:
            torch.save(perception_module.state_dict(), os.path.join(config.models_save_path, 'perception.pth'))
            torch.save(visual_goal_encoder.state_dict(), os.path.join(config.models_save_path, 'visual_goal.pth'))
            torch.save(plan_recognizer.state_dict(), os.path.join(config.models_save_path, 'plan_recognizer.pth'))
            torch.save(plan_proposer.state_dict(), os.path.join(config.models_save_path, 'plan_proposer.pth'))
            torch.save(control_policy.state_dict(), os.path.join(config.models_save_path, 'control_policy.pth'))
            torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))

if __name__ == '__main__':
    from model_config import get_model_args
    config = get_model_args()
    torch.manual_seed(config.seed)
    
    train(config)