clear; clc;

addpath('../../../../Downloads/npy-matlab-master/npy-matlab');

% seen_target = readNPY('seen_target.npy');
% seen_features = readNPY('seen_features.npy');
% seen_att = readNPY('seen_att_one_hot_unique.npy');
% 
% unseen_target = readNPY('unseen_target.npy');
% unseen_features = readNPY('unseen_features.npy');
% unseen_att = readNPY('unseen_att_one_hot_unique.npy');
% 
% att = readNPY('att_spherical.npy');
% 

test_seen_loc = readNPY('test_seen_loc.npy');
test_unseen_loc = readNPY('test_unseen_loc.npy');
train_loc = readNPY('train_loc.npy');
trainval_loc = readNPY('trainval_loc.npy');
val_loc = readNPY('val_loc.npy');
att = readNPY('att.npy');
% 
% z1features = (squeeze(readNPY('../features_vggish.npy')))';
% z2features = (squeeze(readNPY('../features_my.npy')))';
% 
% features = [z1features; z2features];
% labels = readNPY('labels.npy');