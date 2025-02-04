from gym_lattice.envs import Lattice2DEnv
from gym import spaces
import numpy as np
import time

from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

total_arr = []
for i in range(5):
	seq = 'HHPPHHHH' # Our input sequence
	seq = seq.upper()
	env = Lattice2DEnv(seq)

	# Instantiate the agent
	model = DQN('MlpPolicy', env, verbose=1,
	        exploration_fraction=0.2, exploration_final_eps=0.1)
	# Train the agent

	start = time.time()
	model.learn(total_timesteps=int(2e6))
	end = time.time()
	# Save the agent
	# model.save("dqn_lattice")
	# del model  # delete trained model to demonstrate loading

	# # Load the trained agent
	# # NOTE: if you have loading issue, you can pass `print_system_info=True`
	# # to compare the system on which the model was trained vs the current one
	# # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
	# model = DQN.load("dqn_lattice", env=env)

	# Evaluate the agent
	# NOTE: If you use wrappers with your environment that modify rewards,
	#       this will be reflected here. To evaluate with original rewards,
	#       wrap environment in a "Monitor" wrapper before other wrappers.
	mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

	print(mean_reward, std_reward)

	# Enjoy trained agent
	obs = env.reset()
	while not env.done:
	    action, _states = model.predict(obs, deterministic=True)
	    obs, rewards, dones, info = env.step(action)
	    env.render()
	    if dones:
	        print("Episode finished! Reward: {} | Collisions: {} | Actions: {}".format(rewards, info['collisions'], info['actions']))
	        break


	print(f"Total time to train is {end-start}")
	total_arr.append(end-start)

with open(f'{seq}.npy', 'wb') as f:
	np.save(f, np.asarray(total_arr))