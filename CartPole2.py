import gym
import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.agents import TRPOAgent
#from tensorforce import Configuration


NUM_GAMES_TO_PLAY = 70000
MAX_MEMORY_LEN = 100000
CLIP_ACTION = False

env = gym.make("CartPole-v0")
# Create a Proximal Policy Optimization agent
#actions=dict(type='float', shape=(2,)),

agent = PPOAgent(
    states=dict(type='float32', shape=(4,)),
    actions=dict(type='int', num_actions=2),
    network=[
        dict(type='dense', size=32,activation='relu'),
        dict(type='dense', size=32,activation='relu'),
        dict(type='dense', size=32,activation='relu')
    ],

    update_mode=dict(
        unit='episodes',
        batch_size=4,
        frequency=4
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=50000
    ),
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-2
    ),
    subsampling_fraction=0.3,
    optimization_steps=20

)


'''
    batching_capacity=200,
    step_optimizer=dict(
        type='adadelta',
        learning_rate=1e-3)
'''

allRewards = np.zeros(shape=(1,1))

for game in range(NUM_GAMES_TO_PLAY):
    obs = env.reset()
    gameTotalReward = 0
    for step in range(1000):
        env.render()
        a = agent.act(obs)
        #print("ACTION ->",a)
        if CLIP_ACTION:
            for i in range (np.alen(a)):
                if a[i] < -1: a[i]=-0.99999999999
                if a[i] > 1: a[i] = 0.99999999999
        obs, reward, done, info = env.step(a)
        #reward = reward/100
        gameTotalReward = gameTotalReward + reward
        allRewards = np.vstack((allRewards, np.array([reward])))
        if done:
            agent.observe(reward=reward, terminal=True)
        else:
            agent.observe(reward=reward, terminal=False)

        #print("Action: {} Observations Size:{} score: {}".format(a,obs.shape,reward))
        if done:
            print("#",game," last game average ", (gameTotalReward/step)*100,"recent avg", allRewards.flatten()[-10000:].mean()*100,"global avg", allRewards.mean()*100, "max",allRewards.max()*100,"min",allRewards.min()*100 , "steps", step)
            if np.alen(allRewards) >= MAX_MEMORY_LEN:
                allRewards = allRewards[1200:]
            break
