clc; clear all;

% Set payload
payload = single(0.4);

% Set number of covers
numCover = 5;

% Start embedding
embedStart = tic;
for i = 1:numCover
    file_name = [num2str(i), '.pgm'];
    coverPath = fullfile('images_cover', file_name);

    stego = S_UNIWARD(coverPath, payload);

    stegoPath = fullfile('images_stego', file_name);
    imwrite(uint8(stego), stegoPath);
end
embedDuration = toc(embedStart);
fprintf('The duration of embedding: %.2fseconds\n', embedDuration);

% Start extraction
extractStart = tic;

% % Specify images of cover for extraction
% image_set = cell(1, numCover);
% for i = 1:numCover
%     file_name = [num2str(i), '.pgm'];
%     coverPath = fullfile('images_cover', file_name);
%     image_set(i) = {coverPath};
% end
% 
% % Run SRM extraction
% F_cover = SRM(image_set);
% 
% % Resize the features of cover
% C = [];
% Ss = fieldnames(F_cover);
% for Sid = 1:length(Ss)
%     Fsingle = eval(['F_cover.' Ss{Sid}]);
%     C = [C, Fsingle];
% end

% Specify images of stego for extraction
image_set = cell(1, numCover);
for i = 1:numCover
    file_name = [num2str(i), '.pgm'];
    stegoPath = fullfile('images_stego', file_name);
    image_set(i) = {stegoPath};
end

% Run SRM extraction
F_stego = SRM(image_set);

% Resize the features of stego
S = [];
Ss = fieldnames(F_stego);
for Sid = 1:length(Ss)
    Fsingle = eval(['F_stego.' Ss{Sid}]);
    S = [S, Fsingle];
end

extractDuration = toc(extractStart);
fprintf('The duration of extraction: %.2fseconds\n', extractDuration);

file_name = ['S_', num2str(pl), '.mat'];
SPath = fullfile('SRM_features', file_name);
save(SPath, 'S')



% PRNG initialization with seed 1
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));

% Division into training/testing set (half/half & preserving pairs)
random_permutation = randperm(size(C,1));
training_set = random_permutation(1:round(size(C,1)/2));
testing_set = random_permutation(round(size(C,1)/2)+1:end);

% Prepare training features
TRN_cover = C(training_set,:);
TRN_stego = S(training_set,:);

% Prepare testing features
TST_cover = C(testing_set,:);
TST_stego = S(testing_set,:);

% Start training in a ensemble classifier
[trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego);

figure(1);
clf;plot(results.search.d_sub,results.search.OOB,'.b');hold on;
plot(results.optimal_d_sub,results.optimal_OOB,'or','MarkerSize',8);
xlabel('Subspace dimensionality');ylabel('OOB error');
legend({'all attempted dimensions',sprintf('optimal dimension %i',results.optimal_d_sub)});
% title('Search for the optimal subspace dimensionality');

% plot the OOB progress with the increasing number of base learners (at the
% optimal value of subspace dimensionality).
figure(2);
clf;plot(results.OOB_progress,'.-b');
xlabel('Number of base learners');ylabel('OOB error')
% title('Progress of the OOB error estimate');

% Test the performance of S-UNIWARD at the optimal setting
testing_errors = zeros(1,10);
settings = struct('d_sub',2200,'L', 125, 'verbose',2);
for seed = 1:10
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));
    random_permutation = randperm(size(C,1));
    training_set = random_permutation(1:round(size(C,1)/2));
    testing_set = random_permutation(round(size(C,1)/2)+1:end);
    TRN_cover = C(training_set,:);
    TRN_stego = S(training_set,:);
    TST_cover = C(testing_set,:);
    TST_stego = S(testing_set,:);
    [trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego,settings);
    test_results_cover = ensemble_testing(TST_cover,trained_ensemble);
    test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
    false_alarms = sum(test_results_cover.predictions~=-1);
    missed_detections = sum(test_results_stego.predictions~=+1);
    num_testing_samples = size(TST_cover,1)+size(TST_stego,1);
    testing_errors(seed) = (false_alarms + missed_detections)/num_testing_samples;
    fprintf('Testing error %i: %.4f\n',seed,testing_errors(seed));
end
fprintf('---\nAverage testing error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors),std(testing_errors));

figure(3)
plot(x, E_OOB, '-b')
hold on;
plot(x, E_OOB, 'or', 'MarkerSize', 8)
xlabel('Payload');ylabel('E_O_O_B')
