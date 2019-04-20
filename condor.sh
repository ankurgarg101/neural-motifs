universe = vanilla
Executable = /lusr/bin/bash
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "NN Experiment"
Requirements = TARGET.GPUSlot && InMastodon
getenv = True
request_GPUs = 1
request_memory = 15000
+GPUJob = true
Notification = complete
Notify_user = abhinav@cs.utexas.edu


Log = /u/abhinav/Projects/condor_gnlp/logs/motifnet_sgcls_1p.log
Error = /u/abhinav/Projects/condor_gnlp/logs/motifnet_sgcls_1p.err
Output = /u/abhinav/Projects/condor_gnlp/logs/motifnet_sgcls_1p.out
Initialdir = /u/abhinav/Projects/neural-motifs
Arguments =  scripts/train_models_gqa.sh -save_dir /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/motifnet_sgcls/ -nepoch 50 -lr 1e-3
Queue 1

# Arguments = image/scripts/task_2_bs_2.sh oiweoji joasd 
# Queue 1

# Arguments = image/scripts/task_2_bs_2.sh adsjk asdjk asdj
# Queue 1
