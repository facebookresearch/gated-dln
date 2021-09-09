python3 -u four_path_model.py -m \
setup.git.issue_id=1 \
experiment.task_one.name=odd_even \
experiment.task_one.transform=default,invert \
experiment.task_two.name=greater_than_four \
experiment.task_two.transform=default,invert \
experiment=four_path_model \
model.num_layers=3,4 \
model.hidden_layer_cfg.dim=128,512,1024,2048 \
model.should_use_non_linearity=True,False \
model.encoder_cfg.should_share=False \
model.hidden_layer_cfg.should_share=True \
model.decoder_cfg.should_share=False \
setup.script_id=1-1 \
model.weight_init.should_do=True \
model.weight_init.gain=1.0,0.1,0.01,0.001,0.0001 \
model.weight_init.bias=1.0,0,0.1,0.01,0.001 \
optimizer=adam \
optimizer.lr=0.0001 \
'setup.viz.params=[experiment.task_one.name,experiment.task_one.transform,experiment.task_two.name,experiment.task_two.transform,experiment.name,model.num_layers,model.hidden_layer_cfg.dim,model.should_use_non_linearity,model.encoder_cfg.should_share,model.hidden_layer_cfg.should_share,model.decoder_cfg.should_share,model.weight_init.should_do,model.weight_init.gain,model.weight_init.bias,optimizer.lr,optimizer._target_]' \
'setup.description=experiment.task_one.name-${experiment.task_one.name}----experiment.task_one.transform-${experiment.task_one.transform}----experiment.task_two.name-${experiment.task_two.name}----experiment.task_two.transform-${experiment.task_two.transform}----experiment.name-${experiment.name}----model.num_layers-${model.num_layers}----model.hidden_layer_cfg.dim-${model.hidden_layer_cfg.dim}----model.should_use_non_linearity-${model.should_use_non_linearity}----model.encoder_cfg.should_share-${model.encoder_cfg.should_share}----model.hidden_layer_cfg.should_share-${model.hidden_layer_cfg.should_share}----model.decoder_cfg.should_share-${model.decoder_cfg.should_share}----model.weight_init.should_do-${model.weight_init.should_do}----model.weight_init.gain-${model.weight_init.gain}----model.weight_init.bias-${model.weight_init.bias}----optimizer.lr-${optimizer.lr}----optimizer._target_-${optimizer._target_}'