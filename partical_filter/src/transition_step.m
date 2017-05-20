function particles = transition_step(particles, stds)
    % Sample particles from gaussian distribution N(particles, stds)
    % Input:
    % particles:  a matrix of 4 rows and n_particles cols
    % stds: a 4 dimention vector, each is a standard deviation for a
    % dimension of particle
    % Ouput:
    % particles: output particles
    [n_rows,n_particles] = size(particles);
    for i=1:n_rows
        particles(i,:) = particles(i,:) + normrnd(0,stds(i),[1,n_particles]);
    end
end
