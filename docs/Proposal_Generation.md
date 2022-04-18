# QUICK RUN: ACTIONFORMER-PROPOSAL GENERATION


## 1. Extract TSP Features Round 1, run:
  
```
./reproduce_scripts/detector/extract_tsp.sh checkpoints/round1_tsp_62.5.pth tsp_features/round1/
```
**Note:** The tsp features are saved at: tsp_features/round1/

## 2. Train Round 1: Using A1 dataset, run:
  
```
./scripts/train_actionformer.sh configs/track3/track3_actionformer_round1.yaml
```

## 3. Generate Pseudo-A2-v1, run:
  
```
./scripts/gen_pseudo.sh checkpoints/round1_map_27.57.pth.tar
```

## 4. Extract TSP Features Round 2:
  
```
./reproduce_scripts/detector/extract_tsp.sh checkpoints/round1_tsp_62.5.pth tsp_features/round2/
```

## 5. Train Round 2: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v1), run:
```
./scripts/train_actionformer.sh configs/track3/track3_actionformer_round2.yaml
```
**Note:** We use result of Round 2 for the submission.

## 6. Evaluate on A2 dataset, run:
```
./scripts/val_actionformer.sh  checkpoints/epoch_050_map_31.55.pth.tar result_round2.json
```

## 7. Generate submission file, run:
```
./scripts/gen_submit.sh result_round2.json
```