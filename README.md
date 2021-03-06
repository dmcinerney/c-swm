command to remember:

python train.py --dataset /home/jered/Documents/data/c-swm/shapes_train.h5 --encoder small --name shapes_gess --decoder --embedding-dim 32 --batch-size 512 --gess --epochs 200 --eval_dataset /home/jered/Documents/data/c-swm/shapes_eval.h5 --sigma .5 --learning-rate 1e-3

Acheived 99 percent ranking accuracy after 10 steps! Trick: make gaussian noise before decoder and tune sigma correctly.  Latent dimension doesn't look great, but performance is great and I suspect experiments will show that negative sampling fails in other cases where this succeeds.

# Contrastive Learning of Structured World Models

This repository contains the official PyTorch implementation of:

**Contrastive Learning of Structured World Models.**  
Thomas Kipf, Elise van der Pol, Max Welling.  
http://arxiv.org/abs/1911.12247

![C-SWM](c-swm.png)

**Abstract:** A structured understanding of our world in terms of objects, relations, and hierarchies is an important component of human cognition. Learning such a structured world model from raw sensory data remains a challenge. As a step towards this goal, we introduce Contrastively-trained Structured World Models (C-SWMs). C-SWMs utilize a contrastive approach for representation learning in environments with compositional structure. We structure each state embedding as a set of object representations and their relations, modeled by a graph neural network. This allows objects to be discovered from raw pixel observations without direct supervision as part of the learning process. We evaluate C-SWMs on compositional environments involving multiple interacting objects that can be manipulated independently by an agent, simple Atari games, and a multi-object physics simulation. Our experiments demonstrate that C-SWMs can overcome limitations of models based on pixel reconstruction and outperform typical representatives of this model class in highly structured environments, while learning interpretable object-based representations.

## Requirements

* Python 3.6 or 3.7
* PyTorch version 1.2
* OpenAI Gym version: 0.12.0 `pip install gym==0.12.0`
* OpenAI Atari_py version: 0.1.4: `pip install atari-py==0.1.4`
* Scikit-image version 0.15.0 `pip install scikit-image==0.15.0`
* Matplotlib version 3.0.2 `pip install matplotlib==3.0.2`

## Generate datasets

**2D Shapes**:
```bash
python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesEval-v0 --fname data/shapes_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Opposite**:
```bash
python data_gen/env.py --env_id ShapesOppositeTrain-v0 --fname data/shapes_opposite_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesOppositeEval-v0 --fname data/shapes_opposite_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Cursor**:
```bash
python data_gen/env.py --env_id ShapesCursorTrain-v0 --fname data/shapes_cursor_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesCursorEval-v0 --fname data/shapes_cursor_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Cursor + Immovable**:
```bash
python data_gen/env.py --env_id ShapesCursorImmovableTrain-v0 --fname data/shapes_cursor_immovable_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesCursorImmovableEval-v0 --fname data/shapes_cursor_immovable_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Immovable**:
```bash
python data_gen/env.py --env_id ShapesImmovableTrain-v0 --fname data/shapes_imm_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesImmovableEval-v0 --fname data/shapes_imm_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Immovable Fixed**:
```bash
python data_gen/env.py --env_id ShapesImmovableFixedTrain-v0 --fname data/shapes_imm_fixed_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesImmovableFixedEval-v0 --fname data/shapes_imm_fixed_eval.h5 --num_episodes 10000 --seed 2
```

**2D Shapes + Immovable Fixed + No Immovable Actions**:
```bash
python data_gen/env.py --env_id ShapesImmovableFixedTrain-v0 --fname data/shapes_imm_fixed_noa_train.h5 --num_episodes 1000 --seed 1 --no-immovable-actions
python data_gen/env.py --env_id ShapesImmovableFixedEval-v0 --fname data/shapes_imm_fixed_noa_eval.h5 --num_episodes 10000 --seed 2 --no-immovable-actions
```

**3D Cubes**:
```bash
python data_gen/env.py --env_id CubesTrain-v0 --fname data/cubes_train.h5 --num_episodes 1000 --seed 3
python data_gen/env.py --env_id CubesEval-v0 --fname data/cubes_eval.h5 --num_episodes 10000 --seed 4
```

**Atari Pong**:
```bash
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2
```

**Space Invaders**:
```bash
python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_train.h5 --num_episodes 1000 --atari --seed 1
python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_eval.h5 --num_episodes 100 --atari --seed 2
```

**3-Body Gravitational Physics**:
```bash
python data_gen/physics.py --num-episodes 5000 --fname data/balls_train.h5 --seed 1
python data_gen/physics.py --num-episodes 1000 --fname data/balls_eval.h5 --eval --seed 2
```

## Run model training and evaluation

**2D Shapes**:
```bash
python train.py --dataset data/shapes_train.h5 --encoder small --name shapes
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes --num-steps 1
```

**2D Shapes + Cursor**:
```bash
python train.py --dataset data/shapes_cursor_train.h5 --encoder small --name shapes_cursor --action-dim 8 --copy-action --num-objects 6
python eval.py --dataset data/shapes_cursor_eval.h5 --save-folder checkpoints/shapes_cursor --num-steps 1
python vis.py --dataset data/shapes_cursor_eval.h5 --save-folder checkpoints/shapes_cursor --num-steps 1
```

**2D Shapes + Immovable**:
```bash
python train.py --dataset data/shapes_imm_train.h5 --encoder small --name shapes_imm
python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/shapes_imm --num-steps 1

python train.py --dataset data/shapes_imm_train.h5 --encoder small --name shapes_imm_split --split-mlp
python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/shapes_imm_split --num-steps 1
```

**2D Shapes + Cursor + Immovable**:
```bash
python train.py --dataset data/shapes_cursor_immovable_train.h5 --encoder small --name shapes_cursor_immovable --action-dim 8 --copy-action --num-objects 6
python eval.py --dataset data/shapes_cursor_immovable_eval.h5 --save-folder checkpoints/shapes_cursor_immovable --num-steps 1
```

**3D Cubes**:
```bash
python train.py --dataset data/cubes_train.h5 --encoder large --name cubes
python eval.py --dataset data/cubes_eval.h5 --save-folder checkpoints/cubes --num-steps 1
```

**Atari Pong**:
```bash
python train.py --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong
python eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/pong --num-steps 1
```

**Space Invaders**:
```bash
python train.py --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name spaceinvaders
python eval.py --dataset data/spaceinvaders_eval.h5 --save-folder checkpoints/spaceinvaders --num-steps 1
```

**3-Body Gravitational Physics**:
```bash
python train.py --dataset data/balls_train.h5 --encoder medium --embedding-dim 4 --num-objects 3 --ignore-action --name balls
python eval.py --dataset data/balls_eval.h5 --save-folder checkpoints/balls --num-steps 1
```

### Cite
If you make use of this code in your own work, please cite our paper:
```
@article{kipf2019contrastive,
  title={Contrastive Learning of Structured World Models}, 
  author={Kipf, Thomas and van der Pol, Elise and Welling, Max}, 
  journal={arXiv preprint arXiv:1911.12247}, 
  year={2019} 
}
```
