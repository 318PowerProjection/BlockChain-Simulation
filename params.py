# argument of simulation
max_episode = 50000
exploration_noise = 0.5
noise_attenuation = exploration_noise / float(max_episode/2)
save_interval = 20
state_history = 5
max_action = 1.0
min_action = 0.0001
delta_t = 10
graph_update_interval = 20
numpy_seed = 623
load = True
load_number = 50000

# argument of ddpg
tau = 0.005
gamma = 0.99
capacity = 100000
batch_size = 128
update_iteration = 1
actor_alpha = 0.00005
critic_alpha = 0.0005

