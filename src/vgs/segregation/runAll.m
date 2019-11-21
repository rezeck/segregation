clc; close all; clear all;

EXPERIMENTS = 4;
ITERATIONS = 10000;
ROBOTS = 150;
GROUPS = 15;
MULTIPLIER = 10;

data = zeros(EXPERIMENTS, ITERATIONS);
pool = parpool();
fprintf('Running...\n');
for i = (1:EXPERIMENTS)    
    data(i, :) = aggregation(ROBOTS, GROUPS, ITERATIONS, i, MULTIPLIER);
end
delete(pool);
name = ['data-r', int2str(ROBOTS), 'g', int2str(GROUPS), '.mat'];
save(name, 'data');
fprintf('Done!');