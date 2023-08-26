## Learning StarCraft AI
  - Supplementing the original mk1 starcraft bot, this AI will use machine learning to alter its approach over time. 

  [view mk1 bot](https://github.com/CD-Mackay/sc_bot_mk1)

## Files
learning_bot_base:  
  - This script runs a simple, non-learning Starcraft Bot capable of defeating all the pre-packaged computer players that come with StarCraftII

learning_bot_simple:
  - The first iteration of a StarCraft bot with evolutionary learning. Started as a simplified version of learning_bot_base, and will expand to include evolutionary learning protocols to improve performance.
  - This model is based off the tutorial provided by Harrison@pythonprogramming.net. Please check out his [tutorial to learn more](https://pythonprogramming.net/starcraft-ii-ai-python-sc2-tutorial/)


sc2_neural_network:
  - The neural network model for running the learning_bot_simple script. inputs training data from thousands of games against medium A.I. opposition, and takes control over decision making in-game. 
  - If you don't wish to run thousands of simulated games on your own machine, Harrison provides the data collected from his own training [here](https://drive.google.com/file/d/1cO0BmbUhE2HsUC5ttQrLQC_wLTdCn2-u/view)


## To-dos
  - Finish training neural network (30 epochs takes alot of time and processing power)