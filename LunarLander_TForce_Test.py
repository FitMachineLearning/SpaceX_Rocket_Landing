import gym
import numpy as np
from tensorforce.agents import PPOAgent,TRPOAgent


NUM_GAMES_TO_PLAY = 70000
CLIP_ACTIONS = True #set to true if you want actions to be clipped between [-1,1]

env = gym.make("LunarLanderContinuous-v2")
print("Observations ", env.observation_space, " Actions",env.action_space)

# Create a Proximal Policy Optimization agent
    #states=dict(type='float32', shape=(8,)),
    #actions=dict(type='float', shape=(2,)),
agent = PPOAgent(
    states=dict(type='float32', shape=(8,)),
    actions=dict(type='float', shape=(2,)),
    network=[
        dict(type='dense', size=32),
        dict(type='dense', size=32),
        dict(type='dense', size=32)
    ],
    batching_capacity=4096,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-2
    )
)

allRewards = np.zeros(shape=(1,1))

for game in range(NUM_GAMES_TO_PLAY):
    obs = env.reset()
    gameTotalReward = 0
    for step in range(1000):
        env.render()
        a = agent.act(obs)
        #print("ACTION ->",a)
        if CLIP_ACTIONS:
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
            print("#",game," last game average ", (gameTotalReward/step)*100,"global avg", allRewards.mean()*100, "max",allRewards.max()*100,"min",allRewards.min()*100 , "steps", step)

            break
