python3 -u k_path_model.py -m \
setup.git.issue_id=36 \
setup.seed=1 \
experiment.num_epochs=10000 \
experiment.save.frequency=100 \
experiment.task.num_input_transformations=40 \
experiment.task.num_classes_in_selected_dataset=10 \
dataloader.train_config.dataloader.batch_size=8 \
dataloader=mnist \
experiment=k_path_model \
model.num_layers=1 \
model.gate_cfg.mode=5_plus_mod_permute,10_plus_mod_permute,15_plus_mod_permute,20_plus_mod_permute,25_plus_mod_permute,30_plus_mod_permute,35_plus_mod_permute \
model.pretrained_cfg.should_use=False \
experiment.task.mode=permute_input_permute_target \
model.hidden_layer_cfg.dim=128 \
model.should_use_non_linearity=False \
model.encoder_cfg.should_share=False \
model.hidden_layer_cfg.should_share=True \
model.decoder_cfg.should_share=False \
setup.script_id=36-1 \
model.weight_init.should_do=True \
model.weight_init.gain=0.01,0.001,0.0001 \
model.weight_init.bias=0 \
optimizer=sgd \
optimizer.lr=0.0001 \
optimizer.momentum=0.9 \
setup.parent_script_id=29 \
'setup.viz.params=[experiment.save.frequency,experiment.num_epochs,experiment.task.num_input_transformations,experiment.task.num_classes_in_selected_dataset,dataloader.train_config.dataloader.batch_size,dataloader.name,experiment,model.num_layers,model.gate_cfg.mode,model.pretrained_cfg.should_use,experiment.task.mode,model.hidden_layer_cfg.dim,model.should_use_non_linearity,model.encoder_cfg.should_share,model.hidden_layer_cfg.should_share,model.decoder_cfg.should_share,setup.script_id,model.weight_init.should_do,model.weight_init.gain,model.weight_init.bias,optimizer._target_,optimizer.lr,optimizer.momentum]' \
'setup.description=experiment.save.frequency-${experiment.save.frequency}----experiment.num_epochs-${experiment.num_epochs}----experiment.task.num_input_transformations-${experiment.task.num_input_transformations}----experiment.task.num_classes_in_selected_dataset-${experiment.task.num_classes_in_selected_dataset}----dataloader.train_config.dataloader.batch_size-${dataloader.train_config.dataloader.batch_size}----dataloader.name-${dataloader.name}----experiment-${experiment}----model.num_layers-${model.num_layers}----model.gate_cfg.mode-${model.gate_cfg.mode}----model.pretrained_cfg.should_use-${model.pretrained_cfg.should_use}----experiment.task.mode-${experiment.task.mode}----model.hidden_layer_cfg.dim-${model.hidden_layer_cfg.dim}----model.should_use_non_linearity-${model.should_use_non_linearity}----model.encoder_cfg.should_share-${model.encoder_cfg.should_share}----model.hidden_layer_cfg.should_share-${model.hidden_layer_cfg.should_share}----model.decoder_cfg.should_share-${model.decoder_cfg.should_share}----setup.script_id-${setup.script_id}----model.weight_init.should_do-${model.weight_init.should_do}----model.weight_init.gain-${model.weight_init.gain}----model.weight_init.bias-${model.weight_init.bias}----optimizer._target_-${optimizer._target_}----optimizer.lr-${optimizer.lr}----optimizer.momentum-${optimizer.momentum}'