clc; clear all; close all;

%numThreads = 4;
%if matlabpool('size') ~= 0 % checking to see if my pool is already open
%    matlabpool close force local;
%end
%matlabpool(numThreads);

% tipo do experimento, robots, grupos.
params = [[1, 50, 5]; [1, 100, 10]; [1, 150, 15]; ...
          [2, 50, 10]; [2, 100, 10]; [2, 150, 10]; ...
          [3, 150, 5]; [3, 150, 10]; [3, 150, 15]];

for i = 1:length(params)
    param = params(i, :);
    output = ['exp' int2str(param(1)) '-' int2str(param(2)) ...
              'r' int2str(param(3)) 'g.mat'];
    segregation_experiments(param(2), param(3), output);
end
disp('finished!');
