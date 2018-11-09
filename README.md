[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Collaboration and Competition

### Introduction

This project works with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the two agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```

## Installation Instruction
#### The README has instructions for installing dependencies or downloading needed files.

Python 3.6 is required. The program requires PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git  
cd deep-reinforcement-learning/python  
pip install .
```

Run the following to create drlnd kernel in ipython so that the right unity environment is loaded correctly  

```python -m ipykernel install --user --name drlnd --display-name "drlnd"```

Pytorch can be installed with the commands recommended in https://pytorch.org/ for the respective OS. Fo example for Conda package installling into a Windows environment with Python 3.6 is: 
'''
conda install pytorch -c pytorch
'''

### Getting Started

Place <mark>report.ipynb</mark> in the folder <mark>p3_collab-compet/</mark> together with the following two files:

1. ddpg_agent.py - contains the DDPG agent code. 
2. model.py - contains Actor and Critic neural network modules classes

The Unity Reacher environment can be downloaded from here: 


- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

Choose the environment suitable for your machine. Unzipping will create another Tennis_XXX folder. For example, if the Tennis Windows 64-bits environment is downloaded, ```Tennis_Windows_x86_64``` will be created. 

Run 
```p3_collab-compet/report.ipynb```

Enter the right path for the Unity Tennis environment in report.ipynb. For example for a folder consisting a Windows  64-bits environemnt is: 

```
env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")

```

Run the remaining cell as ordered in ```report.ipynb``` to train the Actor-Critic DDPG agent. 

