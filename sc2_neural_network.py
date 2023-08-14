import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random

model = Sequential()


## Main convolutional layers
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

## Connected Dense Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

## Output Layer
model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs/stage1')

hm_epochs = 10

def check_data():
   choices = {
      'no_attacks': no_attacks,
      'attack_closest_to_nexus': attack_closest_to_nexus,
      'attack_enemy_structures': attack_enemy_structures,
      'attack_enemy_start': attack_enemy_start
   }

   total_data = 0

   lengths = []
   for choice in choices:
      print("Length of {} is: {}".format(choice, len(choices[choice])))
      total_data+=len(choices[choice])
      lengths.append(len(choices[choice]))

   return lengths


for i in range(hm_epochs):
    current = 0
    increment = 200
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
      print("WORKING ON {}:{}".format(current, current+increment))
      no_attacks = []
      attack_closest_to_nexus = []
      attack_enemy_structures = []
      attack_enemy_start = []

      for file in all_files[current:current+increment]:
         full_path = os.path.join(train_data_dir, file)
         data = np.load(full_path)
         data = list(data)
         for d in data:
            choice = np.argmax(d[0])
            if choice == 0:
               no_attacks.append([d[0], d[1]])
            elif choice == 1:
               attack_closest_to_nexus.append([d[0], d[1]])
            elif choice == 2:
               attack_enemy_structures.append([d[0], d[1]])
            elif choice == 3:
               attack_enemy_start.append([d[0], d[1]])
            

        

