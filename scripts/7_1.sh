python3 -u k_path_model.py -m \
setup.git.issue_id=7 \
experiment.num_epochs=1000 \
model.gate_cfg.mode=one_plus_mod,mod \
experiment.task.num_classes_in_original_dataset=8 \
dataloader.train_config.dataloader.batch_size=128 \
experiment=k_path_model \
model.num_layers=2 \
model.hidden_layer_cfg.dim=2048 \
model.should_use_non_linearity=False \
model.encoder_cfg.should_share=False \
model.hidden_layer_cfg.should_share=True \
model.decoder_cfg.should_share=False \
setup.script_id=7 \
model.weight_init.should_do=True \
model.weight_init.gain=1.0,0.1,0.01,0.001,0.0001 \
model.weight_init.bias=0,0.1,0.01,0.001 \
optimizer=adam \
optimizer.lr=0.001 \
'setup.viz.params=[experiment.num_epochs,model.gate_cfg.mode,experiment.task.num_classes_in_original_dataset,dataloader.train_config.dataloader.batch_size,experiment.name,model.num_layers,model.hidden_layer_cfg.dim,model.should_use_non_linearity,model.encoder_cfg.should_share,model.hidden_layer_cfg.should_share,model.decoder_cfg.should_share,model.weight_init.should_do,model.weight_init.gain,model.weight_init.bias,optimizer._target_,optimizer.lr]' \
'setup.description=experiment.num_epochs-${experiment.num_epochs}----model.gate_cfg.mode-${model.gate_cfg.mode}----experiment.task.num_classes_in_original_dataset-${experiment.task.num_classes_in_original_dataset}----dataloader.train_config.dataloader.batch_size-${dataloader.train_config.dataloader.batch_size}----experiment.name-${experiment.name}----model.num_layers-${model.num_layers}----model.hidden_layer_cfg.dim-${model.hidden_layer_cfg.dim}----model.should_use_non_linearity-${model.should_use_non_linearity}----model.encoder_cfg.should_share-${model.encoder_cfg.should_share}----model.hidden_layer_cfg.should_share-${model.hidden_layer_cfg.should_share}----model.decoder_cfg.should_share-${model.decoder_cfg.should_share}----model.weight_init.should_do-${model.weight_init.should_do}----model.weight_init.gain-${model.weight_init.gain}----model.weight_init.bias-${model.weight_init.bias}----optimizer._target_-${optimizer._target_}----optimizer.lr-${optimizer.lr}'