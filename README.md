


## running
before trainin should prcess files using process_files_gmde/store_graph_properties.py
it generates a folder inside the folder of each dataset that includes the graphicaö geatures

running is similar to RotatE and MDE_adv , plus a folder that incldues node features in the path:  -node_feat_path




#### sample run

with best hyperparameters on wn18rr 

 CUDA_VISIBLE_DEVICES=2 python run.py --do_train --do_test -save ../experiments/kge_baselines_wn18rr --data_path ../data/WN18RR  --model MDE  -n 624 -b 210 -d 300 -g 2.5 -a 2.5 -adv -lr .0005 --max_steps 250000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ../data/WN18RR/train_node_features --cuda -psi 14.0 
 
 in cas it does not fit in memory reduce -n and id not the -d and reduce the --max_steps if trains fast
 
-----------


#### note:
running is without --triples_are_mapped when runned without dic files are generated in process_datasets/convert_files.py 
 