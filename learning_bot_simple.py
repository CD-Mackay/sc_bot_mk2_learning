import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, CYBERNETICSCORE, GATEWAY, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, OBSERVER, ZEALOT 
import random
import time
import cv2
import numpy as np
from examples.terran.proxy_rax import ProxyRaxBot
import keras
import math

HEADLESS = False

class r2_sc2(sc2.BotAI):
    def __init__(self, use_model=False):
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_stargate,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.attack_known_enemy_unit,
                        11: self.attack_known_enemy_structure,
                        12: self.expand,
                        13: self.do_nothing,
                        }
        # self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        self.scouts_and_spots = {}

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-STAGE1")

    def on_end(self, game_result):
        print('---on_end called---')
        print(game_result, self.use_model)


        with open("log.txt","a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))
                
        if game_result == Result.Victory:
          np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data, dtype=object))



    async def on_step(self, iteration):
        # self.iteration = iteration
        self.time = (self.state.game_loop / 22.4) / 60
        await self.distribute_workers()
        await self.scout()
        await self.intel()
        await self.do_something()

    
    async def do_something(self):
        
        if self.time > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 14)
            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))
            
            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped])

    async def scout(self):
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)
        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)
        
        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15
        
        assign_scout = True
        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False
        
        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                              location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                              active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                              if location not in active_locations:
                                  if unit_type == PROBE:
                                      for unit in self.units(PROBE):
                                          if unit.tag in self.scouts_and_spots:
                                              continue
                                  
                                  await self.do(obs.move(location))
                                  self.scouts_and_spots[obs.tag] = location
                                  break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

            
          
        # if len(self.units(OBSERVER)) > 0:
        #     scout = self.units(OBSERVER)[0]
        #     if scout.is_idle:
        #         enemy_location = self.enemy_start_locations[0]
        #         move_to = self.random_location_variance(enemy_location)
        #         await self.do(scout.move(move_to))
        
        # else:
        #     for rf in self.units(ROBOTICSFACILITY).ready.idle:
        #         if self.can_afford(OBSERVER) and self.supply_left > 0:
        #             await self.do(rf.train(OBSERVER))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x+= ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y+= ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]
        
        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to
        
    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        draw_dict = {
                     NEXUS: [15, (0, 255, 0)],
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     VOIDRAY: [3, (255, 100, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)]
                    }
        
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_name = ['nexus', 'commandcenter', 'orbitalcommand', 'planetaryfortress', 'hatchery']
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_name:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)

        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_name:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 225), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe", "scv", "drone"]
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)
        
        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0
        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0
        
        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0
        
        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        
        

        self.flipped = cv2.flip(game_data, 0)
        if not HEADLESS:  
          resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
          cv2.imshow('Intel', resized)
          cv2.waitKey(1)

    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.time / 3):
            for rf in self.units(ROBOTICSFACILITY).ready.idle:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))
    
    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready
        cybercores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybercores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))
        
        if not cybercores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)
    
    async def build_workers(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_pylons(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
          if self.can_afford(PYLON):
            await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))
    
    
    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def expand(self):
      if self.units(NEXUS).amount < self.time/2 and self.can_afford(NEXUS):
        await self.expand_now()

    async def offensive_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < self.time:
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)
            
    async def build_offensive_force(self):

      for sg in self.units(STARGATE).ready.idle:
          if self.can_afford(VOIDRAY) and self.supply_left > 0:
              await self.do(sg.train(VOIDRAY))
    
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]
        
    async def attack(self):
      if len(self.units(VOIDRAY).idle) > 0:
          target = False
          if self.time > self.do_something_after:
              if self.use_model:
                  prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                  choice = np.argmax(prediction[0])
                  choice_dict = {
                      0: "no attack",
                      1: "attack close to nexus",
                      2: "attack enemy structure",
                      3: "attack enemy start"
                  }
                  print("Choice #{}:{}".format(choice, choice_dict[choice]))
              else:
                choice = random.randrange(0,4)
              if choice == 0:
                  ## No attack
                  wait = random.randrange(7, 100) / 100
                  self.do_something_after = self.time + wait
              
              elif choice == 1:
                  ## attack enemy unit closest to nexus
                  if len(self.known_enemy_units) > 0:
                      target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
              
              elif choice == 2:
                  ## Attack enemy structures
                  if len(self.known_enemy_structures) > 0:
                      target = random.choice(self.known_enemy_structures)
              
              elif choice == 3:
                  ## Attack known enemy start location
                  target = self.enemy_start_locations[0]

              if target:
                  for vr in self.units(VOIDRAY):
                      await self.do(vr.attack(target))
              
              y = np.zeros(4)
              y[choice] = 1
              print("y:", y)
              self.train_data.append([y, self.flipped])

counter = 0

while counter < 5:
    
  run_game(maps.get("AbyssalReefLE"), [
      Bot(Race.Protoss, r2_sc2(use_model=True)),
      Computer(Race.Terran, Difficulty.Easy)
  ], realtime=False)
  counter+=1 
