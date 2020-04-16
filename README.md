# DeepContrast
Nonlinear control of adaptive optics for high contrast imaging using deep neural networks

## Prerequisites
DeepContrast makes use of [HCIPy](https://github.com/ehpor/hcipy), an open-source object-oriented framework written in Python for performing end-to-end
simulations of high-contrast imaging instruments (Por et al. 2018). Please see the link for installation instructions.  
[OpenAI Gym](https://gym.openai.com/) is needed for the HCI Testbench Markov Decision Process environment.

## Code
There are several files in this repository which I have used for various experiments, but the main code for the DeepContrast controller is in the root directory. The notebook [HCI_DRL_2Dstate.ipynb](./HCI_DRL_2Dstate.ipynb) is the main notebook in which I run the simulation to train a deep reinforcement learning agent for AO contrast control. This makes use of the [model.py](./model.py), [agent.py](./agent.py), [noise_model.py](./noise_model.py), [replay_buffer.py](./replay_buffer.py) files.

## Description
The simulation is run in HCIPy, and the agent is trained using the [Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1509.02971.pdf) algorithm, in order to maximize contrast within a specific dark hole. This dark hole is generated with the help of an APP coronagraph found in the [coronagraphs](./coronagraphs) folder and is maintained dark under residual atmospheric turbulence with AO. The reinforcement learning agent is coded in [TensorFlow 2.0](https://www.tensorflow.org/).
