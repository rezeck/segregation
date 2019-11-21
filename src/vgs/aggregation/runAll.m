clc; close all; clear all;

EXPERIMENTS = 100;
ITERATIONS = 100000;
ROBOTS = 150;
MULTIPLIER = 100;

fprintf('Running...\n');
pool = parpool();
for GROUPS = [15, 10, 5]
    data = zeros(EXPERIMENTS, ITERATIONS);        
    parfor i = (1:EXPERIMENTS)    
        data(i, :) = aggregation(ROBOTS, GROUPS, ITERATIONS, i, MULTIPLIER);
    end    
    name = ['data-r', int2str(ROBOTS), 'g', int2str(GROUPS), '.mat'];
    save(name, 'data');
end
delete(pool);
fprintf('Done!');
