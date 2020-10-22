import os
import numpy
import torch

from models import *
#from render_browser import render_browser

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

#@render_browser
def test_experiment(config):
    deg = config.deg()
    env = deg.get_env()
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    with torch.no_grad():
        perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
        visual_goal_encoder = VisualGoalEncoder(config.visual_state_dim, config.goal_dim).to(device)
        plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
        control_module = ControlModule(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)

        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))

        obvs = env.reset()
        for i in range(config.n_test_evals):
            obvs = env.reset()
            goal = torch.from_numpy(deg.get_random_goal()).float()
            goal = goal.reshape(1, goal.shape[2], goal.shape[0], goal.shape[1])
            goal = perception_module(goal)
            goal, _, _, _ = visual_goal_encoder(goal) 
            t, done = 0, False

            while (not done) and t <= config.max_test_timestep: 
                # TODO : Figure out way to set done tru when goal is reached
                visual_obv, dof_obv = torch.from_numpy(deg._get_obs(obvs, deg.vis_obv_key).float(), torch.from_numpy(deg._get_obs(obvs, deg.dof_obv_key).float()
                visual_obv = visual_obv.reshape(1, visual_obv.shape[2], visual_obv.shape[0], visual_obv.shape[1])
                dof_obv = dof_obv.reshape(1, dof_obv.shape[0])
                state = perception_module(visual_obv, dof_obv)
                z_p, _, _, _ = plan_proposer(state, goal)
                
                action, _ = control_module.step(state, goal, z_p)
                obvs, _, done, _ = env.step(action[0])
                #yield env.render(mode='rgb_array')
                env.render()
                t += 1

if __name__ == '__main__':
    from model_config import get_model_args
    config = get_model_args()
    test_experiment(config)