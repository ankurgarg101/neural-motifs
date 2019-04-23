universe = vanilla
Executable = /lusr/bin/bash
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "NN Experiment"
Requirements = TARGET.GPUSlot && InMastodon
getenv = True
request_GPUs = 1
+GPUJob = true
Notification = complete
Notify_user = abhinav@cs.utexas.edu


Log = /scratch/cluster/ankgarg/gqa/temp_abhinav/motifnet_sgcls_1p.log
Error = /scratch/cluster/ankgarg/gqa/temp_abhinavmotifnet_sgcls_1p.err
Output = /scratch/cluster/ankgarg/gqa/temp_abhinav/motifnet_sgcls_1p.out
Initialdir = /scratch/cluster/ankgarg/gqa/neural-motifs/
Arguments =  scripts/train_models_gqa.sh -save_dir /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/motifnet_sgcls/ -nepoch 50 -lr 1e-3
Queue 1

# Arguments = image/scripts/task_2_bs_2.sh oiweoji joasd 
# Queue 1

# Arguments = image/scripts/task_2_bs_2.sh adsjk asdjk asdj
# Queue 1
