clc; clear;

addpath('npy-matlab');

features = readNPY('mel.npy');
% features = squeeze(features);
% imagesc(features)