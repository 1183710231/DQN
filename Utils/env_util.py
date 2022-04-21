from Environments.environment import Environment


def get_env_info(env_id):
    env = Environment(env_id)
    env.load()
    # wkr修改，采用大小为2000的子图
    # env.load_child_gragh()
    dim_state = env.dim_state
    # num_actions = env.num_ents * env.num_rels + 1
    # wkr修改
    num_actions = env.num_ents + 1
    ent_dict=env.ent_dict

    return env, dim_state, num_actions,ent_dict
