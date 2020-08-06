clc; clear;

addpath('../../../Downloads/npy-matlab-master/npy-matlab');

features = readNPY('mel.npy');
% features = squeeze(features);
% imagesc(features)