% main file of Tracking
% clc;
clear;
close all;

%% Parameter initialization
dataset = 'David'; % 'car' or 'David'
sz_I = [56, 56]; % width, height
if strcmp(dataset, 'car')
    data_dir = '../data/car/imgs/';
    save_dir = '../data/car/results/';
    ini_rect = [63 50 116 91]';% Initial position [x y width height]
    suffix = 'jpg';
elseif strcmp(dataset, 'David')
    data_dir = '../data/David2/imgs/';
    save_dir = '../data/David2/results/';
    ini_rect = [127, 55, 56, 56];
    suffix = 'jpg';
end

% initial particle, [c_x, c_y, s_x, s_y], (c_x, c_y) is the center
% location, s_x and s_y are scales compared with sz_I
ini_particle = convert_rect_2_particle(ini_rect, sz_I);
n_particles = 400; % Number of particles
stds = [4, 4, 0.01, 0.01]; % standard deviation of [c_x, c_y, s_x, s_y]
feature_type = 'HOG'; % The default feature, you may use another
% feature if you want, see feature_extract function

%% process
% the second parameter is 'jpg' for car and png for bolt
[n_frames, s_frames] = readImageSequences(data_dir, suffix);
img = imread(s_frames{1});
current_rect = ini_rect;
current_particle = ini_particle;% state = particle
% tracked_rect stores rectangles from t = 0  to t = T
tracked_rect = zeros(4, n_frames);
tracked_rect(:, 1) = current_rect;
particles = repmat(current_particle, 1, n_particles); %current state -> 400 particles
% y is the representation of image in last tracked rect
n_templates=5;
templates(:,1) =feature_extract(img, current_rect, sz_I, feature_type); % 15*15 -> 255 vec  
templates_weights(1,1)=1;
show_and_store(tracked_rect, 1, img, s_frames, save_dir)

for t = 2:n_frames%3
    tic
    % "Transition" step, sample particles from gaussian model N(particles, stds)
    particles = transition_step(particles, stds);
    img = imread(s_frames{t});

    % "Weighting" step, compute weights of particles, inside this function,
    % you need to use feature_extract function to extract features of
    % particles and use compute_similarity function to compute similarity
    % between particles and last tracked rect
    % Note: weights should be normalized to sum to 1
    weights = weighting_step(img, particles, sz_I, templates,templates_weights, feature_type);

    % choose particle with largest weight and compute feature of it
    [curr_weight, idx_max] = max(weights);
    current_particle = particles(:, idx_max);
    current_rect = convert_particle_2_rect(current_particle, sz_I);
    tracked_rect(:, t) = current_rect;
    
    % update templates
    curr_templates = feature_extract(img, current_rect, sz_I, feature_type);
    [templates,templates_weights]=update_templates(curr_templates,templates,n_templates);
    templates_weights
%     if cmp_weights(curr_weight,templates_weights)
%        templates(:,end) =curr_templates;
%        templates_weights(1,end)=curr_weight;
%        templates_weights=templates_weights/sum(templates_weights);
%        [templates_weights,sort_idxs]=sort(templates_weights,'descend');
%        templates=templates(:,sort_idxs);
%     end
    
    % show tracked rect
    show_and_store(tracked_rect, t, img, s_frames, save_dir);

    % "Resample" step
    particles = resample_step(particles, weights);
    toc
end
tracked_rect
close all;