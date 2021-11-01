from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
# from train_PEBBLE import Workspace
from train_PEBBLE import main_wrapper
import argparse
import os

# import hydra

# @hydra.main(config_path='../config/trian_PEBBLE_quadruped_100.yaml') #, strict=True)
# def main(cfg):
#     workspace = Workspace(cfg)
#     workspace.run()

def train(doodad_config, variant):
    # args = argparse.Namespace()
    # d = vars(args)
    # for key, val in variant.items():
    #     d[key] = val
    # Workspace(args)
    # main()
    main_wrapper()
    save_doodad_config(doodad_config)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", help="path to the config file directory", required=True)
    # parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory", required=True)
    # parser.add_argument("--dry", action='store_true', help="dry run, local no doodad")
    # args = parser.parse_args()

    # if not os.path.isabs(args.path):
    #     raise ValueError('experiment path must be absolute!')

    params_to_sweep = {}
    # default_params = {
    #     'env': 'quadruped_walk',
    #     'seed': 12345,
    #     'gradient_update': 1,
    #     'activation': 'tanh',
    #     'num_seed_steps': 1000,
    #     'num_unsup_steps': 9000,
    #     'num_train_steps': 1000000,
    #     'replay_buffer_capacity': 1000000,
    #     'num_interact': 30000,
    #     'max_feedback': 1000,
    #     'reward_lr': 0.0003,
    #     'reward_batch': 100,
    #     'reward_update': 50,
    #     'feed_type': 1,
    #     'teacher_beta': 1,
    #     'teacher_gamma': 1,
    #     'teacher_eps_mistake': 0,
    #     'teacher_eps_skip': 0,
    #     'teacher_eps_equal': 0,
    #     'actor_lr': 0.0001,
    #     'critic_lr': 0.0001,
    #     'agent': 'sac',
    #     'experiment': 'PEBBLE',
    #     'segment': 50,
    #     'reset_update': 100,
    #     'topK': 5,
    #     'ensemble_size': 3,
    #     'device': 'cuda',
    #     'large_batch': 10,
    #     'label_margin': 0.0,
    #     'reward_schedule': 0,
    #     'eval_frequency': 10000,
    #     'num_eval_episodes': 10,
    #     'log_frequency': 10000,
    #     'log_save_tb': 'true',
    #     'save_video': 'false',
    # }
    default_params = {}

    # mode = 'here_no_doodad'
    mode = 'azure'
    sweep_function(
        train,
        params_to_sweep,
        default_params=default_params,
        log_path='exp_quadruped',
        mode=mode,
        use_gpu=True,
        num_gpu=1,
    )


