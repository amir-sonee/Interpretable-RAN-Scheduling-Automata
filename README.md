# Interpretable-RAN-Scheduling-Automata-Learning

Code for symbolic learning of automata using ILASP to abstract the resource scheduling task in RAN presented in [[`Amir Sonee et al., PIMRC 2026`]](https://pimrc2026.ieee-pimrc.org/technical-sessions).

### 1. [Installation](#1-installation) 
### 2. [Training the automaton using ILASP](#2-training-the-automaton-using-ilasp) 
### 3. [Testing automaton](#3-testing-automaton)
### 4. [Comparison with baselines](#4-comparison-with-baselines)

## 1. Installation
This code runs on Linux or MacOS systems with Python 3. Please download repository with the following command:

```bash
git clone https://github.com/amir-sonee/Interpretable-RAN-Scheduling-Automata.git
```
### Python packages

```bash

cd Interpretable-RAN-Scheduling-Automata
pip install -r requirements.txt
```

To avoid the conflict of your current system installations, please be advised of creating a [`virtual environement`](https://docs.python.org/3/tutorial/venv.html) like ```venv``` and then proceed to installing the dependancies. 

In case of getting the error ```SDL.h file not found``` when installing the packages, then first install ```SDL2``` based on the operating system as given in [[Install dependencies]](https://github.com/ertsiger/induction-subgoal-automata-rl#install-additional-dependencies): 

- Ubuntu: ```sudo apt-get install libsdl2-dev```
- MacOS: ```brew install sdl sdl_image sdl_mixer sdl_ttf portmidi```

### ILASP and Clingo
Please download the binaries for installing ILASP system (inductive logic programming of answer set programs) from [`ILASP-releases`](https://github.com/ilaspltd/ILASP-releases/releases) and cling from [`cling`](https://github.com/potassco/clingo/releases). Then, copy the ```ILASP``` and ```clingo``` binaries into the ```bin``` folder. You can also directly install them using the ```install_binaries.sh``` in [[induction-subgoal-automata-rl]](https://github.com/ertsiger/induction-subgoal-automata-rl/blob/master/install_binaries.sh). 

```bash
cd Interpretable-RAN-Scheduling-Automata-Learning
./install_binaries.sh
```

## 2. Training the automaton using ILASP

The automaton can be learned using the already developed Inductive Logic learner of Answer Set Programs (ILASP) by running ```run_ilasp``` using the following script 
```bash
cd ProbIRM
source venv/bin/activate
cd rm_marl/rm_learning/ilasp
python3 run_ilasp.py datasets/config_RAN.json task_RAN solution_RAN graph_RAN -f results_RAN -s bfs
```

where 

- ```config_RAN``` is a ```json``` file containing the configuration of the automaton for learning which includes number of states, set of observables/symbols or propositions, training dataset of three classes of labelled traces (goal, dead-end and incomplete), and the setting of the inductive logic reasoning. For the case of our paper we considered two sets of propositions to learn the rules for temporal abstraction in scheduling task.   
  ```
   "num_states": 6,
   "max_disjunction_size": 1,
   "learn_acyclic": true,
   "use_compressed_traces": true,
   "avoid_learning_only_negative": true,
   "prioritize_optimal_solutions": false,
   "observables":["voip0","voip1","voip2","voip3", "bvs1", "bvs2", "bvs3", "ftp0", "ftp1", "ftp3", "ftp4", "nr"],
   "goal_examples":
   "deadend_examples":
   "inc_examples":
  ```
- ```task_RAN``` is the file created by ILASP system for learning the task
- ```solution_RAN``` is the ILASP solution file containing the rules of transition between edges of the automaton's states
-  ```graph_RAN``` is the ```pdf``` file showing the graph of automaton
-  ```results_RAN``` is the folder containing the above three files. 

## 3. Testing the learned automaton

Run the ```test_ilasp.py``` in the ```test``` folder of ```ilasp```:

```bash
cd test
python3 test_ilasp.py ../datasets/test_traces.json ../results_RAN/solution_RAN
```

where ```test_traces.json``` is the file containing the testing traces and ```../results_RAN/solution_RAN``` is the solution file of the learning task by ILASP.

The results of the predictions are saved automatically as given in the test file along with a separate file including the false negative test traces for simplicity and a separate file for confusion matrix image as well as other scores like Macro and F1 scores. 

## 4. Comparison with baselines

In the directory of ```baselines```, compile the files written separately for train and test of different models including ```Transformer```, ```LSTM```,  ```RNN```, ```decision_tree``` and ```random_forest``` on the same train and test datasets used for automaton. In the following scripts, first argument is the model file for training and testing like ```transformer4admission.py```, the second argument ```config_RAN.json``` is the file containing training dataset and the third argument ```test_traces.json``` is the test file. In case the test file is stored in the directory of “test-ProbISA” then you need to follow the below instruction.

```bash
    python3 transformer.py ../datasets/config_RAN.json ../datasets/test_traces.json
	python3 LSTM.py ../datasets/config_RAN.json ../datasets/test_traces.json
	python3 RNN.py ../datasets/config_RAN.json ../datasets/test_traces.json 
```
