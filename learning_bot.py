import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

class r2_sc2(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()


run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, r2_sc2()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)