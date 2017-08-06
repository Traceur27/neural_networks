import gym
import random
import numpy as np
import keras

from statistics import mean, median
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

TRAINING_DATA_FILENAME = 'training_data_cart_pole2.npy'

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 10000


def play_random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

#play_random_games();

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        
        for _ in range(goal_steps):
            action = random.randrange(0, 2) # take one of random actions available in this env
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # converto to one-hot encoding
                if data[1] == 1:
                    output = [0, 1]
                else:
                    output = [1,0]
                # append observation and action taken
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save(TRAINING_DATA_FILENAME, training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data_save

def neural_network_model(input_size):
    model = Sequential()
    model.add(Dense(units=128, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    #model.add(Dense(units=256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.8))

    #model.add(Dense(units=512))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.8))

    #model.add(Dense(units=256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.8))

    #model.add(Dense(units=128))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.8))

    model.add(Dense(units=2))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


#initial_population()
training_data = initial_population() # np.load(TRAINING_DATA_FILENAME)
print(training_data.shape)
print(training_data[0][0].shape)
print(training_data[0][1])

X = np.array([i[0] for i in training_data]).reshape(len(training_data), 4)
y = np.array([i[1] for i in training_data]).reshape(len(training_data), 2)
model = neural_network_model(len(X[0]))
model.fit(X, y, epochs=3)

scores = []
choices = []

for game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            prediction = model.predict(prev_obs.reshape(1, len(prev_obs)))
            action = np.argmax(model.predict(prev_obs.reshape(1, len(prev_obs))))
        choices.append(action)

        new_obs, reward, done, info = env.step(action)
        prev_obs = new_obs
        game_memory.append([new_obs, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average score', sum(scores)/len(scores))
print('Choice 1', choices.count(1)/len(choices), 'Choice 0', choices.count(0)/len(choices))
