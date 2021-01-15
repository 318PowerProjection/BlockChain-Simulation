# argument of simulation
max_episode = 5000
exploration_noise = 0.5
noise_attenuation = 0.0025
save_interval = 5
state_history = 5
max_action = 1.0
min_action = 0.0001
delta_t = 1
graph_update_interval = max_episode / 100

# argument of ddpg
tau = 0.005
gamma = 0.99
capacity = 1e5
batch_size = 128
update_iteration = 10
actor_alpha = 0.00005
critic_alpha = 0.0005

