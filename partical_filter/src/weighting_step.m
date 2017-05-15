function weights = weighting_step(img, particles, sz_I, templates,templates_weights, feature_type)
    % This function first compute feature for each particle, then compute
    % similarity between features of particles and y

    % Input:
    % img: input image
    % particles: a 4xN matrix, each col corresponds to a particle state
    % sz_I: base size that a rect should be resized to
    % y: the feature of the last tracked rect.
    % feature_type: the type of feature
    % Oputput:
    % weights: a vector, each element corresponds to a particle

    % get particles_feature 
    [n_states,n_particles] = size(particles);
    [n_features,n_templates]=size(templates);
    particles_feature = zeros( n_features , n_particles ); 
%     tic
    parfor i = 1:n_particles
        now_rect = convert_particle_2_rect( particles(:,i) , sz_I );
        now_feature = feature_extract(img, now_rect, sz_I, feature_type);
        particles_feature(:,i) = now_feature;
    end
%     toc
    % get similarity to y
    similarity = compute_similarity(particles_feature,templates);
    for i=1:n_templates
       similarity(i,:)= similarity(i,:)*templates_weights(1,i);
    end
    similarity=sum(similarity,1);
    % Note: normalize!
    weights = similarity / sum(similarity);
end
