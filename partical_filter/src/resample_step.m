function particles = resample_step(particles, weights,method)
    % resaple particles according to weights
    % Input: 
    % particles: a 4xN matrix, each column is a particle
    % weights: a vector, each element corresponds to a particle
    % Output:
    % particles: resampled particles
    if nargin<3
       method ='systematic';
    end
    
    n_particles=length(particles);
    switch method
        case 'systematic'
            edges=[0 cumsum(weights)];
            edges=min(edges,1);
            edges(end)=1;
            edges=sort(edges);
            start=rand()/n_particles;
            [~,idx]=histc(start:1/n_particles:1,edges); 
        case 'multinomial'
            idx=randsample(1:n_particles,n_particles,true,weights);
            
        otherwise 
            disp('unknwon method') 
    end
    particles=particles(:,idx);
end