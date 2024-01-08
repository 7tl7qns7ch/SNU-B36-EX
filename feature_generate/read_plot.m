clear; clc;
addpath('../../../Downloads/npy-matlab-master/npy-matlab');
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
% 
% test_seen_loc = readNPY('test_seen_loc.npy');
% test_unseen_loc = readNPY('test_unseen_loc.npy');
% train_loc = readNPY('train_loc.npy');
% trainval_loc = readNPY('trainval_loc.npy');
% val_loc = readNPY('val_loc.npy');
% att = readNPY('att.npy');

% features = (readNPY('features.npy'))';
% labels = readNPY('labels.npy');
resultsroot = 're_8';
file = dir(resultsroot);

cd(resultsroot);
sn = [500];
nz = [13];
gh = [1024];
i = 1;
j = 1;

k = 1;

% sn = [500];
% nz = [9, 25, 57, 121];
% gh = [1024];

% for i = 1:length(sn)
%     for j = 1:length(nz)
%         for k = 1:length(gh)
%             file_name = strcat('sn', num2str(sn(i)), 'nz', num2str(nz(j)), 'gh', num2str(gh(k)));
%             cd(file_name)
%             c_err{i, j, k} = readNPY('c_err.npy');
%             h_acc{i, j, k} = readNPY('h_acc.npy');
%             loss_D{i, j, k} = readNPY('loss_D.npy');
%             loss_G{i, j, k} = readNPY('loss_G.npy');
%             seen_acc{i, j, k} = readNPY('seen_acc.npy');
%             unseen_acc{i, j, k} = readNPY('unseen_acc.npy');
%             wasserstein_dist{i, j, k} = readNPY('wasserstein_dist.npy');
%             cd ..
%         end
%     end
% end
% 
% h_acc = cell2mat(reshape(h_acc, [], 1)');
% loss_D = cell2mat(reshape(loss_D, [], 1)');
% loss_G = cell2mat(reshape(loss_G, [], 1)');
% figure; plot(h_acc);
% figure; plot(loss_D); hold on; plot(loss_G);

% imagesc(h_acc);
% colormap('jet');
file_name1 = strcat('sn', num2str(sn(i)), 'nz', num2str(nz(j)), 'gh', num2str(gh(k)));

for i = 3:length(file)
    file_name = file(i).name;
    
    cd(file_name)
%     cd(file_name1)
    c_err{i-2} = readNPY('c_err.npy');
    h_acc{i-2} = readNPY('h_acc.npy');
    loss_D{i-2} = readNPY('loss_D.npy');
    loss_G{i-2} = readNPY('loss_G.npy');
    seen_acc{i-2} = readNPY('seen_acc.npy');
    unseen_acc{i-2} = readNPY('unseen_acc.npy');
    wasserstein_dist{i-2} = readNPY('wasserstein_dist.npy');
%     cd ..
    cd ..
    
end
h_acc = cell2mat(h_acc);
h_acc_mean = mean(h_acc);
figure; plot(h_acc(:, 7));
xlabel('epochs * 10');
ylabel('accuracy')
hold on;

[M, I] = max(h_acc, [], 1);
%M = reshape(M, [3, 3])';

seen_acc = cell2mat(seen_acc);
plot(seen_acc(:, 7));
unseen_acc = cell2mat(unseen_acc);
plot(unseen_acc(:, 7));

legend('harmonic mean', 'seen accuracy', 'unseen accuracy')

loss_D = cell2mat(loss_D);
figure; plot(loss_D(:, 7));
xlabel('epochs * 10');
ylabel('loss');

loss_G = cell2mat(loss_G);
hold on; plot(loss_G(:, 7));
legend('loss_D', 'loss_G');

wasserstein_dist = cell2mat(wasserstein_dist);
figure; plot(wasserstein_dist(:, 7));
xlabel('epochs * 10');
ylabel('wasserstein distance');

for i = 1:9
    S(i) = seen_acc(I(i), i);
    U(i) = unseen_acc(I(i), i);
end
%S = reshape(S, [3, 3])';
%U = reshape(U, [3, 3])';

acc = zeros(3, 9);
acc(1, 1) = S(2);
acc(1, 2) = S(8);
acc(1, 3) = S(5);

acc(1, 4) = S(1);
acc(1, 5) = S(7);
acc(1, 6) = S(4);

acc(1, 7) = S(3);
acc(1, 8) = S(9);
acc(1, 9) = S(6);

acc(2, 1) = U(2);
acc(2, 2) = U(8);
acc(2, 3) = U(5);

acc(2, 4) = U(1);
acc(2, 5) = U(7);
acc(2, 6) = U(4);

acc(2, 7) = U(3);
acc(2, 8) = U(9);
acc(2, 9) = U(6);

acc(3, 1) = M(2);
acc(3, 2) = M(8);
acc(3, 3) = M(5);

acc(3, 4) = M(1);
acc(3, 5) = M(7);
acc(3, 6) = M(4);

acc(3, 7) = M(3);
acc(3, 8) = M(9);
acc(3, 9) = M(6);