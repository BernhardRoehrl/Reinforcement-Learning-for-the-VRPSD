import numpy as np
import gym
import random
import time
from IPython import display


env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 15000        # Total episodes
max_steps_per_episode = 99  # Max steps per episode

learning_rate = 0.09      # Learning rate
discount_rate = 0.95        # Discounting rate

# Exploration parameters
exploration_rate = 1.0                 # Exploration rate
max_exploration_rate = 1.0             # Exploration probability at start
min_exploration_rate = 0.0001            # Minimum exploration probability
exploration_decay_rate = 0.002             # Exponential decay rate for exploration prob

# List of rewards
rewards_all_episodes = []

# 2 For life or until learning is stopped
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # 3. Choose an action a in the current world state (s)
        # First we randomize a number
        exploration_rate_threshold = random.uniform(0, 1)

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])

        rewards_current_episode += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done:
            break

    # Reduce epsilon (because we need less and less exploration)
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("\n*********Average reward per thousand episodes******\n")
for r in rewards_per_thousand_episodes:
    print(count, ":", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n**********Q-Table**************\n", q_table)

print(env.action_space.sample())

"""for episode in range(3):
    state = env.reset()
    done = False
    print("******EPISODE ", episode+1, "******\n\n\n\n")
    time.sleep(0.3)

    for step in range(max_steps_per_episode):
        display.clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[new_state, :])
        new_state, reward, done, info = env.step(action)
        print("\n\n\n***************new_state, reward, done, info =", new_state, reward, done, info, "***************")

        if done:
            display.clear_output(wait=True)
            env.render()
            if reward ==1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                display.clear_output(wait=True)
                break

            state = new_state

env.close()"""
print(env.observation_space.low)

