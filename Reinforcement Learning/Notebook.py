# pip install gym
# pip install gym[toy_text]

import gym 
import random
env = gym.make('Taxi-v3', render_mode='ansi')

env.reset()
print(env.render())


# Tabela P
env.reset()
state = env.encode(3,1,2,0) # (Taxi row, Taxi Colmun)
print("Estado: ", state)
print(env.P[state])

###Resolvendo sem RL
env.reset()
epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done,truncate, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Episódios necessários: {}".format(epochs))
print("Penalidades sofridas: {}".format(penalties))


### mostrando resultado
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.2)
        
print_frames(frames)


# Treinamento da tabela T
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# Busca indice com valor mais alto
np.argmax(np.array([3, 5]))

# 1-10% 3-90%
# exploration / exploitation
# 0 = south 1 = north 2 = east 3 = west 4 = pickup 5 = dropoff

from IPython.display import clear_output

alpha = 0.1 # Aprendizagem
gamma = 0.6 # Desconto do valor futuro
epsilon = 0.1 # Decisão aleatória

for i in range(100000):
  estado,info = env.reset()

  penalidades, recompensa = 0, 0
  done = False
  while not done:
    # Exploração
    if random.uniform(0, 1) < epsilon:
      acao = env.action_space.sample()
    # Exploitation
    else:
      acao = np.argmax(q_table[estado])

    proximo_estado, recompensa, done,truncate, info = env.step(acao)

    q_antigo = q_table[estado, acao]
    proximo_maximo = np.max(q_table[proximo_estado])

    q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
    q_table[estado, acao] = q_novo

    if recompensa == -10:
      penalidades += 1

    estado = proximo_estado

  if i % 100 == 0:
    clear_output(wait=True)
    print('Episódio: ', i)

print('Treinamento concluído')

q_table[346]

## Avaliação

total_penalidades = 0
episodios = 50
frames = []

for _ in range(episodios):
  estado,info = env.reset()
  penalidades, recompensa = 0, 0
  done = False
  while not done:
    acao = np.argmax(q_table[estado])
    estado, recompensa, done,truncate, info = env.step(acao)

    if recompensa == -10:
      penalidades += 1
    
    frames.append({
        'frame': env.render(),
        'state': estado,
        'action': acao,
        'reward': recompensa
    })

  total_penalidades += penalidades

print('Episódios', episodios)
print('Penalidades', total_penalidades)

print_frames(frames)

