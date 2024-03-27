function [nets, infos] = spdnet_afew_new(varargin)
%set up the path
confPath; % Upload the path of some toolkits to the workspace

% Base directory for blocks
baseDir = fullfile('./data/afew/UAV_CONV');

% Find all block directories
blockDirs = dir(fullfile(baseDir, 'block_*'));
blockDirs = blockDirs([blockDirs.isdir]);

nets = cell(1, numel(blockDirs));
infos = cell(1, numel(blockDirs));

for i = 1:numel(blockDirs)
    opts.dataDir = fullfile(baseDir, blockDirs(i).name);
    % Modify imdbPathtrain to reflect the block-specific .mat file name
    blockNumber = regexp(blockDirs(i).name, 'block_(\d+)', 'tokens', 'once');
    if isempty(blockNumber)
        error(['Unable to extract block number from directory name: ', blockDirs(i).name]);
    end
    % Use the extracted block number to construct the path to the .mat file for the current block
    opts.imdbPathtrain = fullfile(baseDir, sprintf('sample_for_SPDNet_UAV_%s.mat', blockNumber{1}));
    
    opts.batchSize = 30; % Original is 30
    opts.test.batchSize = 1;
    opts.numEpochs = 3; % Maximum number of epochs
    opts.gpus = [];
    opts.learningRate = 0.01 * ones(1,opts.numEpochs); % Learning rate
    opts.weightDecay = 0.0005;
    opts.continue = 1;

    % SPDNet initialization
    net = spdnet_init_afew_deep_v1();

    % Load metadata for the current block
    if exist(opts.imdbPathtrain, 'file')
        load(opts.imdbPathtrain);
    else
        error(['Data file for block ', blockNumber{1}, ' not found.']);
    end

    % SPDNet training for the current block
    [net, info] = spdnet_train_afew(net, spd_train, opts);

    % Store results for each block
    nets{i} = net;
    infos{i} = info;

    % Assuming infos is an array/cell of info structures from each block
    averageAccuracies = arrayfun(@(x) x.averageAccuracy, infos);
    
    % Save to file
    save(fullfile(baseDir, 'average_accuracies.mat'), 'averageAccuracies');


end
