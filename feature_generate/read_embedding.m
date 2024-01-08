clc; clear;

addpath('../../../Downloads/npy-matlab-master/npy-matlab');
count = 0

file = dir('re_8_old');

cd re_8_old/G_split_D_split_ca;
count = 0
countcount = 0;

file_name = 'logits_seen.npy';
b = [];
b = readNPY(file_name);
best = 15;
bb = squeeze(b(best, :, :));
[n, ~] = size(bb);
aaa = [];

for j = 1:n
    count = count + 1
    ex_b = exp(bb(j, :));
    ex_b = ex_b / sum(ex_b);
    ex_b = double(ex_b);
    a(count, :) = ex_b;
    aaa = [aaa;ex_b];
    if mod(j, 10) == 0
        countcount = countcount + 1;
        aa(countcount, :) = mean(aaa, 1);
        aaa = [];
    end
    
end

file_name = 'logits_unseen.npy';
b = [];
b = readNPY(file_name);
best = 15;
bb = squeeze(b(best, :, :));
[n, ~] = size(bb);
aaa = [];

for j = 1:n
    count = count + 1
    ex_b = exp(bb(j, :));
    ex_b = ex_b / sum(ex_b);
    ex_b = double(ex_b);
    a(count, :) = ex_b;
    aaa = [aaa;ex_b];
    if mod(j, 50) == 0
        countcount = countcount + 1;
        aa(countcount, :) = mean(aaa, 1);
        aaa = [];
    end
    
end

unseen_set = [6;7;8;17;22;24;27;29;47;48;67;72;83;93;107;111;115;118;120;126;129;134;136;138;140;144;154;155;171;175;176;177;178];
seen_set = [1;2;3;4;5;9;10;11;12;13;14;15;16;18;19;20;21;23;25;26;28;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;49;50;51;52;53;54;55;56;57;58;59;60;61;62;63;64;65;66;68;69;70;71;73;74;75;76;77;78;79;80;81;82;84;85;86;87;88;89;90;91;92;94;95;96;97;98;99;100;101;102;103;104;105;106;108;109;110;112;113;114;116;117;119;121;122;123;124;125;127;128;130;131;132;133;135;137;139;141;142;143;145;146;147;148;149;150;151;152;153;156;170;172;173;174;179;180;181;182];
[nn, mm] = size(aa);
output = zeros(nn, mm);
for i = 1:length(seen_set)
    output(seen_set(i), :) = aa(i, :);
end

for i = 1:length(unseen_set)
    output(unseen_set(i), :) = aa(i + length(seen_set), :);
end
    
% for i = 3:4
%     file_name = file(i).name;
%     b = [];
%     b = readNPY(file_name);
%     [n, m] = size(b);
%     aaa = [];
% 
%     for j = 1:n
%         count = count + 1
%         ex_b = exp(b(j, :));
%         ex_b = ex_b / sum(ex_b);
%         ex_b = double(ex_b);
%         a(count, :) = ex_b;
%         if i == 3
%             aaa = [aaa;ex_b];
%             if mod(j, 10) == 0
%                 countcount = countcount + 1;
%                 aa(countcount, :) = mean(aaa, 1);
%                 aaa = [];
%             end
%         elseif i == 4
%             aaa = [aaa;ex_b];
%             if mod(j, 50) == 0
%                 countcount = countcount + 1;
%                 aa(countcount, :) = mean(aaa, 1);
%                 aaa = [];
%             end
%         end            
% 
%     end
% 
% end
    
% for i = 0:51
%     file_name = strcat('logits_sp/logits_', num2str(i), '.npy');
%     b = [];
%     b = readNPY(file_name);
%     [n, m] = size(b);
%     
%     for j = 1:n
%         count = count + 1
%         ex_b = exp(b(j, :));
%         ex_b = ex_b / sum(ex_b);
%         a(count, :) = ex_b;
%     end
%         
% end

% countcount = 0;
% for i = 1:50:count
%     countcount = countcount + 1;
%     aa(countcount, :) = mean(a(i:i+49, :), 1);
% end

% load att_splits.mat

% c = reshape(a(56, :), [5, 3, 13]);
count2 = 0;
att = zeros(3, 195);
for i = 1:5
    for j = 1:3
        for k = 0:12
            count2 = count2 + 1;
            att(1, count2) = i;
%             if att(1, count2) == 1
%                 atty(1, count2) = 'MB';
%             elseif att(1, count2) == 2
%                 atty(1, count2) = 'HD';
%             elseif att(1, count2) == 3
%                 atty(1, count2) = 'HH';
%             elseif att(1, count2) == 4
%                 atty(1, count2) = 'CD';
%             elseif att(1, count2) == 5
%                 atty(1, count2) = 'VC';
%             end                
            att(2, count2) = j;
            att(3, count2) = k;
        end
    end
end

map = zeros(100, 3);
for i = 1:100
    map(i, :) = [1, 0.9501 - 0.0095 * i, 0.9501 - 0.0095 * i];    
end
for i = 1:30:169
    c = output(i, :);
    figure;
    scatter3(att(1, :), att(3, :), att(2, :), 65, c, 'filled');
    ax = gca;
    text()
    view(28, 14);
    xlabel('Type', 'FontSize', 14, 'FontWeight', 'bold');
    xticklabels({'MB', 'HD', 'HH', 'CD', 'VC'});
    ylabel('Distance', 'FontSize', 14, 'FontWeight', 'bold');
    zlabel('Floor', 'FontSize', 14, 'FontWeight', 'bold');
    zticklabels([1], {' '}, [2], {' '}, [3]);

    cb = colorbar;
%     cb.Label.String = 'Probability';
%     cb.FontSize = 11;
    colormap(map);
end
    
