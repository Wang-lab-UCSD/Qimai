#!/bin/bash
# when running "sub_generate_dataset_with_embedding.job" on "1000_1000_per_exp_train_6mer.csv", it submitted 103 jobs, however we 
# only got 88 files in the end. 15 jobs failed silently somehow. When we checked the log, we didn't see any error. So we decided
# to find out those unfinished jobs and resubmit them

# got finished job indices
cd /new-stg/home/cong/DPI/dataset/Encode3/embeddings_512bp
ll 1000_per_exp_train_6mer_* | awk '{gsub(/.*[/]|[.].*/, "", $0)} 1' | awk -F'[_]' '{print $NF}' | sort -n > ../tmp.txt

# substract 1 for each line in tmp.txt
awk '{print $0-1}' < tmp.txt > num_finished_jobs.txt

# compare and got unfinished job indices
cd ../
comm -23 <(sort num_files.txt) <(sort num_finished_jobs.txt) | sort -n > num_unfinished_jobs.txt